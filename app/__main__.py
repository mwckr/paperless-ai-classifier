"""`python -m app` starts the service."""
import uvicorn

from .config import Settings


def main() -> None:
    s = Settings.load()
    uvicorn.run("app.main:app", host=s.API_HOST, port=int(s.API_PORT), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
