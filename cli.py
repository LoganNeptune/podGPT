#!/usr/bin/env python3
import click
import torch
from podGPT import BigramLanguageModule, encode, decode  # Adjust the import if needed

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('prompt', nargs=-1)
def interactive(prompt):
    """
    Start an interactive dialogue with the podGPT model.
    Use the syntax: podGPT -- Your initial prompt here.
    Everything after '--' will be taken as the initial prompt.
    """
    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Running on {device}")

    # Load the trained model (make sure 'podGPT_entire_model.pt' is in the same folder)
    try:
        model = torch.load("podGPT_entire_model.pt", map_location=device)
    except FileNotFoundError:
        click.echo("Model file not found. Please ensure 'podGPT_entire_model.pt' is present.")
        return
    model.to(device)
    model.eval()

    # Use the provided prompt arguments, or start with an empty conversation if none provided.
    current_context = " ".join(prompt) if prompt else ""
    if current_context:
        click.echo(f"Initial prompt: {current_context}")
    else:
        click.echo("No initial prompt provided. Starting a blank dialogue.")

    click.echo("Entering interactive mode. Press Ctrl-C to exit.\n")
    # Interactive loop: continuously accept input and generate response.
    try:
        while True:
            # Read user input
            user_input = input("You: ")
            # If the user types something, append it to the conversation context.
            if user_input.strip():
                current_context += " " + user_input.strip()
            else:
                # If no new input, leave context unchanged.
                pass

            # Encode the current conversation into tokens.
            tokens = encode(current_context)
            context_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

            # Generate the next 50 tokens (adjust max_new_tokens as appropriate)
            generated_tokens = model.generate(context_tensor, max_new_tokens=50)[0].tolist()
            generated_text = decode(generated_tokens)

            # Append the generated output to the conversation context.
            current_context += " " + generated_text

            # Output the model’s response.
            print("podGPT:", generated_text)
    except KeyboardInterrupt:
        print("\nExiting interactive mode. Goodbye!")

if __name__ == '__main__':
    interactive()