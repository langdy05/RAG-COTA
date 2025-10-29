# ## Stage 1: Use an official, lightweight Python image
# This starts with a clean, minimal version of Linux that includes Python 3.11.
FROM python:3.11-slim-bookworm

# ## Stage 2: Set a working directory
# This creates a folder inside the container where your app will live.
WORKDIR /app

# ## Stage 3: Copy dependency files
# We copy these first because they change less often. Docker can cache this step,
# making future builds much faster.
COPY pyproject.toml poetry.lock ./

# ## Stage 4: Install Poetry and dependencies
# This installs Poetry itself, then uses your lock file to install the exact
# versions of all your project's libraries.
RUN pip install poetry
RUN poetry install --no-root

# ## Stage 5: Copy your application code
# This copies the rest of your files (app.py, documents folder, etc.) into the container.
COPY . .

# ## Stage 6: Define the startup command
# This is the command that will run when the container starts. It tells Poetry to
# run the Gunicorn server, binding it to the port provided by the host (like Cloud Run).
CMD ["poetry", "run", "gunicorn", "app:app", "-b", "0.0.0.0:$PORT"]