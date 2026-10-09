use super::*;

#[test]
fn a_database_written_before_native_mode_opens_unchanged() {
    let workspace = tempfile::tempdir().unwrap();
    let path = db_path(workspace.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    {
        // What the store did before: its own connection, its own migrations.
        let conn = Connection::open(&path).unwrap();
        prepare_connection(&conn).unwrap();
        migrations::apply(&conn).unwrap();
    }
    crate::record_session_start(
        workspace.path(),
        "legacy",
        "agent",
        "Agent",
        "legacy",
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        crate::get_session(workspace.path(), "legacy").unwrap().id,
        "legacy"
    );
}

#[test]
fn the_store_shares_the_drivers_connection() {
    let workspace = tempfile::tempdir().unwrap();
    crate::record_session_start(
        workspace.path(),
        "shared",
        "agent",
        "Agent",
        "shared",
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    // Any other native handle on the file is the same connection, with the
    // session pragmas the store applied.
    let native = SqliteNative::open(db_path(workspace.path())).unwrap();
    let (rows, foreign_keys): (i64, i64) = native
        .run_blocking(|conn| {
            let rows = conn.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
            let fk = conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
            Ok::<_, rusqlite::Error>((rows, fk))
        })
        .unwrap()
        .unwrap();
    assert_eq!(rows, 1);
    assert_eq!(foreign_keys, 1);
}

#[test]
fn a_transaction_rolls_back_on_error() {
    let workspace = tempfile::tempdir().unwrap();
    let result: Result<()> = with_transaction(workspace.path(), |conn| {
        conn.execute_batch("CREATE TABLE scratch (n INTEGER); INSERT INTO scratch VALUES (1);")
            .storage_context("scratch")?;
        Err(TinyAgentsError::Storage("abort".into()))
    });
    assert!(result.is_err());
    let exists: i64 = with_connection(workspace.path(), |conn| {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'scratch'",
            [],
            |row| row.get(0),
        )
        .storage_context("probe")
    })
    .unwrap();
    assert_eq!(exists, 0);
}

#[test]
fn an_unopenable_path_is_a_storage_error() {
    let workspace = tempfile::tempdir().unwrap();
    // A file where the `session_db` directory should be.
    std::fs::write(workspace.path().join("session_db"), b"not a dir").unwrap();
    let error = with_connection(workspace.path(), |_| Ok(())).unwrap_err();
    assert!(matches!(error, TinyAgentsError::Storage(_)), "{error:?}");
    assert!(driver_error("x").to_string().contains("session DB"));
}

#[test]
fn a_panicking_call_leaves_the_connection_usable() {
    let workspace = tempfile::tempdir().unwrap();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_transaction(workspace.path(), |conn| {
            conn.execute_batch("CREATE TABLE half (n INTEGER)")
                .storage_context("half")?;
            panic!("session bug mid-transaction");
        })
    }));
    assert!(panicked.is_err(), "the panic still reaches the caller");
    // The lock is not poisoned and the open transaction was rolled back.
    let tables: i64 = with_connection(workspace.path(), |conn| {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'half'",
            [],
            |row| row.get(0),
        )
        .storage_context("probe")
    })
    .unwrap();
    assert_eq!(tables, 0);
    with_transaction(workspace.path(), |_| Ok(())).unwrap();
}
