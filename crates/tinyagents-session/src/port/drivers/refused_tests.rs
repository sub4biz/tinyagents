use super::*;

fn refused() -> Refused {
    Refused {
        kind: ErrorKind::Unavailable,
        message: "down".into(),
    }
}

#[tokio::test]
async fn every_document_call_reports_the_refusal() {
    let docs = refused();
    let kind = |error: StorageError| {
        assert_eq!(error.message(), "down");
        error.kind()
    };
    assert_eq!(docs.capabilities(), Capabilities::default());
    assert_eq!(
        kind(
            docs.ensure_collection(&CollectionSpec::new("c"))
                .await
                .unwrap_err()
        ),
        ErrorKind::Unavailable
    );
    assert!(docs.get("c", "i").await.is_err());
    assert!(
        docs.put("c", "i", Value::Null, Precondition::None)
            .await
            .is_err()
    );
    assert!(docs.delete("c", "i", Precondition::None).await.is_err());
    assert!(docs.query("c", &Query::all()).await.is_err());
    assert!(docs.count("c", &Filter::All).await.is_err());
    assert!(docs.delete_where("c", &Filter::All).await.is_err());
    assert!(
        docs.claim("c", &Filter::All, &[], &Value::Null)
            .await
            .is_err()
    );
    assert!(docs.drop_collection("c").await.is_err());
}

#[tokio::test]
async fn every_stream_call_reports_the_refusal() {
    let streams = refused();
    assert!(streams.append("s", Value::Null).await.is_err());
    assert!(streams.append_batch("s", vec![]).await.is_err());
    assert!(streams.read_window("s", 0, 1).await.is_err());
    assert!(StreamStore::len(&streams, "s").await.is_err());
    assert!(streams.truncate_before("s", 0).await.is_err());
    assert!(streams.delete_stream("s").await.is_err());
    assert!(streams.streams("").await.is_err());
}
