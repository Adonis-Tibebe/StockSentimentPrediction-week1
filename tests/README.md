# Test Suite

## Test Files

- `test_data_loader.py`: Tests for data loading functionality
- `test_data_cleaner.py`: Tests for data cleaning operations
- `test_topic_modeling.py`: Tests for topic modeling functions

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_topic_modeling.py

# Run with verbose output
pytest -v
```

## Test Data

Sample data files in `test_data/` for unit testing:
- Small subset of news headlines
- Known topics and events
- Edge cases for testing

---

## Extending the Test Suite

- Add new test files for additional modules as needed.
- Use pytest fixtures for reusable test data.
- Ensure coverage for edge cases and error handling.

---