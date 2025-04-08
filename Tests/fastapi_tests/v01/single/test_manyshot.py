def test_process_manyshot_categorize(
    client, manyshot_csv, perform_post_request, validate_encoded_response
):
    """
    Tests the POST method for processing categorization from a base64-encoded XLSX file
    to ensure it handles the input correctly and returns the expected response.
    """
    # The endpoint to test
    url = "/cv/v01/labels/"

    # The payload including all necessary fields from RequestCategorization
    payload = {
        "file_encoded": manyshot_csv,
        "extension": ".csv",
        "text_column": "review",  # Example value for text_column
        "ex_label_column": "sentiment",  # Example value for ex_label_column
        "categories": [
            {"name": "positive", "description": "Positive Category"},  # Example category
            {"name": "negative"},  # Another example category
        ],
        "mode": "Evaluation",  # Use 'Evaluation' or 'Production' as Mode
        "model": ["Many Shot"],  # Example model list
        "few_shot_count": "2",  # Example value for few_shot_count
        "many_shot_train_ratio": "0.85",  # Example value for many_shot_train_ratio
    }

    # Make the POST request
    response = perform_post_request(client, url, payload)

    # Assertions to verify the response status and content
    validate_encoded_response(response, "application/json", "encoded_xlsx")
