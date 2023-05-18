from data_preparation.download import download
import pytest
import os


def test_download(tmpdir):
    # Temporary directory to save the downloaded file
    save_dir = str(tmpdir)

    # URL of the file to download (replace with your own URL)
    url = "https://example.com/myfile.txt"

    # File name to save the downloaded file
    fname = os.path.join(save_dir, "myfile.txt")

    # Call the download function
    download(url, fname)

    # Assert that the file has been downloaded
    assert os.path.exists(fname)

    # Assert that the downloaded file size is greater than 0
    assert os.path.getsize(fname) > 0


# Run the test
pytest.main(["-v", "--tb=line", __file__])
