from .app import app


def main():
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)
