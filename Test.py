import os
import fal_client

# Set API key
os.environ["FAL_KEY"] = "a99380f7-6ab1-4a8f-8563-6b7243248676:af61914351d84df4a8d0c3e5765dfad7"

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

result = fal_client.subscribe(
    "fal-ai/flux/dev",
    arguments={
        "prompt": "Imran Khan speech in front of crowd."
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)

print(result)
