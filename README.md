podGPT is an interactive GPT model trained specifically for generating continuous text dialogues.


## Installation & Setup

git clone https://github.com/LoganNeptune/podGPT.git
cd podGPT

Step 2: Install Dependencies

Make sure you have Python installed (3.10+ recommended). Then install dependencies:

pip install torch click



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
