# Security

This application is designed for local, single-machine deployments. It has no login system and stores email app passwords in the local database. Do not expose the web interface to the public internet or to an untrusted network.

The Docker Compose configuration publishes the app on localhost by default. Keep this setting unless you fully understand and accept the risk of exposing the user interface.

Do not commit real `.env` files, project databases, email credentials, camera credentials, model weights, image data, alert outputs or logs to GitHub.
