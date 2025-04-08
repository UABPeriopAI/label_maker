def test_process_categorize_xlsx(
    client, encoded_xlsx, perform_post_request, validate_encoded_response
):
    url = "/cv/v01/labels/"

    payload = {
        "file_encoded": encoded_xlsx,
        "extension": ".xlsx",
        "index_column": "index",
        "text_column": "review",
        "ex_label_column": "sentiment",
        "categories": [
            {"name": "positive", "description": "Positive Category"},
            {"name": "negative"},
        ],
        "mode": "Production",
        "model": ["Zero Shot", "Few Shot"],
        # The line `"ex_label_column": "sentiment"` in the `payload` dictionary is specifying the
        # column name in the Excel file that contains the existing labels or categories for the data.
        # In this case, it indicates that the column named "sentiment" in the Excel file contains the
        # labels or categories that the model will use for training or classification purposes.
        "few_shot_count": "2",
        "many_shot_train_ratio": "0.85",
    }

    # Make the POST request
    response = perform_post_request(client, url, payload)

    # Assertions to verify the response status and content
    validate_encoded_response(response, "application/json", "encoded_xlsx")
