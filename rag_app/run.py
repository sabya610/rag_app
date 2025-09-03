from app import create_app
import argparse


app = create_app()

# ====================
# RUN
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)