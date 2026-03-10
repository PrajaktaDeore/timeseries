from pathlib import Path
import yfinance as yf


def download_btc_usd_to_csv(
    output_dir: str = "projectfolder/data",
    filename: str = "BTC-USD.csv",
    period: str = "max",
    interval: str = "1d",
) -> Path:
    """Fetch BTC-USD data from yfinance and save it as a CSV file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = yf.download("BTC-USD", period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError("No data returned for BTC-USD.")

    output_path = out_dir / filename
    data.to_csv(output_path)
    return output_path


if __name__ == "__main__":
    saved_path = download_btc_usd_to_csv()
    print(f"Saved BTC-USD data to: {saved_path}")
