podGPT is an interactive GPT model trained specifically for generating continuous text dialogues.


## Installation & Setup

git clone https://github.com/LoganNeptune/podGPT.git
cd podGPT

sudo apt install python3-venv python3-pip -y
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install torch
pip install click #inference

⸻

## Running podGPT

# Make the binary executable

chmod +x cli.py
mv cli.py podGPT  # rename for convenience

## Run the model

Start an interactive session:

./podGPT -- Your initial prompt goes here

Example:

./podGPT -- Once upon a time

If no prompt is provided, the model will begin with a blank slate:

./podGPT

Interact with the model by typing your input at the prompt.
Exit the interactive session at any time with Ctrl+C.
