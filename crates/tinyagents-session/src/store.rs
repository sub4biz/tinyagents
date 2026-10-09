//! Connection lifecycle for the session database: path resolution, pragma
//! setup, and the autocommit/transaction entry points every other module in
//! this crate opens the database through.
//!
//! Every caller in `super::ops`, `super::retention`, and `super::run_ledger`
//! goes through [`with_connection`] or [`with_transaction`] rather than
//! opening a [`Connection`] directly, so schema migrations
//! ([`super::migrations`]) and the busy-timeout/WAL/foreign-key pragmas
//! ([`prepare_connection`]) are applied uniformly on every path into the
//! database.
//!
//! The connection is the tinystoragedrivers SQLite driver's (native mode),
//! which runs WAL with `synchronous = NORMAL`: commits survive a process
//! crash, but an OS crash or power loss can roll back the most recent ones
//! (the database stays consistent). See the crate README's durability note.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use rusqlite::Connection;
use tinystoragedrivers_sqlite::SqliteNative;

use super::context::StorageContext;
use super::migrations;
use tinyagents_harness::error::{Result, TinyAgentsError};

/// Process-wide cache of opened session databases, keyed by the database
/// file path.
///
/// The connection itself belongs to the SQLite driver's native mode: one
/// shared connection per file for the whole process, behind one lock, which
/// every other handle the host opens on the same file (async or sync) also
/// goes through. This cache keeps that connection alive and records that the
/// session pragmas and migrations ran on it, so they run once per path rather
/// than on every call — a `Connection::open` per operation was measured as
/// the dominant cost of session-store calls under load.
fn connection_cache() -> &'static Mutex<HashMap<PathBuf, SqliteNative>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, SqliteNative>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Subdirectory of the workspace holding the session database.
const DB_SUBDIR: &str = "session_db";
/// Database filename inside [`DB_SUBDIR`].
const DB_FILE: &str = "sessions.db";

/// How long a statement waits for a competing writer's lock before giving up
/// with `SQLITE_BUSY`.
///
/// # This is a guarantee we own, not a bug fix
///
/// SQLite's own default is zero — a `BEGIN IMMEDIATE` that finds the write lock
/// held would fail instantly rather than wait — and every claim, gate and
/// sequence allocation in this module is written on the assumption that racing
/// writers *serialize* at `BEGIN`.
///
/// That assumption was, as it happens, already satisfied: `rusqlite`'s
/// `Connection::open` calls `sqlite3_busy_timeout(db, 5000)` unconditionally,
/// so the connections here have never actually had a zero timeout. Setting it
/// explicitly changes no behaviour today. It is worth doing anyway, because the
/// alternative is that a load-bearing correctness property of this module is
/// supplied by an undocumented default of a transitive dependency, invisible at
/// every call site and free to change in a patch release. Stating it here makes
/// the dependency deliberate and greppable.
///
/// Five seconds is long enough to ride out any transaction this module takes
/// (all of them are a handful of small statements) and short enough to surface
/// a genuine deadlock rather than hang.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolves the session database path for a workspace root.
///
/// Kept public so hosts can locate the file for backup, inspection, or
/// migration without reproducing the layout.
pub fn db_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(DB_SUBDIR).join(DB_FILE)
}

/// Returns the native handle for `db_path`, opening it and preparing the
/// shared connection (pragmas, then migrations) the first time this path is
/// seen.
///
/// Pragma setup and migrations run exactly once per path, when the handle is
/// created — not on every call — since both are properties of the
/// connection/database, not of an individual operation.
fn cached_connection(db_path: &Path) -> Result<SqliteNative> {
    let mut cache = connection_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = cache.get(db_path) {
        return Ok(existing.clone());
    }

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).storage_context(&format!(
            "failed to create session_db directory: {}",
            parent.display()
        ))?;
    }

    let native = SqliteNative::open(db_path).map_err(|error| {
        TinyAgentsError::Storage(format!(
            "failed to open session DB: {}: {error}",
            db_path.display()
        ))
    })?;
    native
        .run_blocking(|conn| {
            prepare_connection(conn)?;
            migrations::apply(conn)
        })
        .map_err(driver_error)??;

    tracing::debug!(
        target: "tinyagents_session::store",
        path = %db_path.display(),
        "[session] opened session DB on the sqlite driver's native mode"
    );
    cache.insert(db_path.to_path_buf(), native.clone());
    Ok(native)
}

/// A failure of the driver itself (its connection lock), as a storage error.
fn driver_error(error: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Storage(format!("session DB: {error}"))
}

/// Opens (or reuses) the workspace's session database connection, applying
/// schema migrations on first use, and runs `f` against the connection.
///
/// The connection is the SQLite driver's shared one for this file, guarded
/// by its lock, so operations on the same path serialize the way they did
/// when every call opened its own file handle. It blocks the calling thread
/// while it waits for that lock and runs `f`. Note that because the
/// connection is cached rather than reopened, a database file atomically
/// replaced at this same path after the first call will *not* be picked up —
/// the process keeps its original handle.
pub fn with_connection<T>(
    workspace_dir: &Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    let db_path = db_path(workspace_dir);
    let native = cached_connection(&db_path)?;
    // A panic in `f` must not unwind while the driver's connection lock is
    // held: that would poison the one shared connection for every later
    // call in the process. Catch it inside, roll back any transaction it
    // left open, release the lock, and only then resume the panic, so the
    // caller still sees it and the connection stays usable (as the earlier
    // store's poison-tolerant lock allowed).
    let outcome = native
        .run_blocking(|conn| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(conn)));
            if outcome.is_err()
                && !conn.is_autocommit()
                && let Err(rollback) = conn.execute_batch("ROLLBACK")
            {
                tracing::warn!(
                    "[session] rollback after a panicking session call failed: {rollback}"
                );
            }
            outcome
        })
        .map_err(driver_error)?;
    match outcome {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// Applies the per-connection pragmas every session-DB handle needs.
///
/// The driver already opens its connection with WAL and a busy timeout; they
/// are set again here so the session's own guarantees stay stated (and
/// greppable) in this crate rather than inherited silently. `foreign_keys`
/// is per connection and only this crate wants it, so it is set here.
fn prepare_connection(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)
        .storage_context("failed to set session DB busy_timeout")?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA foreign_keys = ON;",
    )
    .storage_context("failed to apply session DB pragmas")?;
    Ok(())
}

/// Opens the session database and runs `f` inside a single **immediate**
/// write transaction, committing on `Ok` and rolling back on `Err`.
///
/// [`with_connection`] hands out an autocommit connection: each statement
/// commits on its own, so a multi-statement read-then-write sequence has no
/// isolation at all. Any operation whose correctness depends on the state it
/// read still holding when it writes — a compare-and-swap claim, a gate that
/// checks dependencies before acting — must use this instead.
///
/// `BEGIN IMMEDIATE` rather than the default deferred begin: it takes the
/// write lock up front, so two racing claims serialize at `BEGIN` instead of
/// discovering the conflict at COMMIT time and failing with `SQLITE_BUSY`
/// after one of them has already decided it won.
pub fn with_transaction<T>(
    workspace_dir: &Path,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    with_connection(workspace_dir, |conn| {
        conn.execute_batch("BEGIN IMMEDIATE")
            .storage_context("begin session DB transaction")?;
        match f(conn) {
            Ok(value) => {
                conn.execute_batch("COMMIT")
                    .storage_context("commit session DB transaction")?;
                Ok(value)
            }
            Err(err) => {
                // Roll back best-effort: the caller's error is the one worth
                // reporting, and a failed rollback (connection already gone)
                // must not mask it.
                if let Err(rollback_err) = conn.execute_batch("ROLLBACK") {
                    tracing::warn!(
                        "[session] rollback after error failed: {rollback_err} (original: {err})"
                    );
                }
                Err(err)
            }
        }
    })
}

#[cfg(test)]
pub fn with_memory_connection<T>(f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
    let conn =
        Connection::open_in_memory().storage_context("failed to open in-memory session DB")?;
    prepare_connection(&conn)?;
    migrations::apply(&conn)?;
    f(&conn)
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
