import os

from openai import OpenAI

# Set up your OpenAI API key

client = OpenAI()


def paraphrase_caption(caption):
    """
    Sends the caption to OpenAI API to get a paraphrased version.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or use gpt-4 if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Paraphrase the following caption while keeping the meaning intact: {caption}"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        paraphrased_caption = response.choices[0].message.content.strip()
        return paraphrased_caption
    except Exception as e:
        print(f"Error paraphrasing caption: {e}")
        return None

def process_caption_file(folder_path):
    """
    Reads the original caption, paraphrases it, and saves the result.
    """
    original_caption_path = os.path.join(folder_path, "bim_caption_attack_caption.txt")
    paraphrased_caption_path = os.path.join(folder_path, "bim_caption_attack_paraphrased.txt")
    
    # Check if the original caption file exists
    if not os.path.exists(original_caption_path):
        print(f"No caption file found in {folder_path}. Skipping.")
        return
    
    # Read the original caption
    with open(original_caption_path, 'r') as file:
        original_caption = file.read().strip()
    # Get the paraphrased caption
    paraphrased_caption = paraphrase_caption(original_caption)
    
    if paraphrased_caption:
        # Save the paraphrased caption to a new file
        with open(paraphrased_caption_path, 'w') as file:
            file.write(paraphrased_caption)
        print(f"Saved paraphrased caption in {paraphrased_caption_path}")
    else:
        print(f"Failed to paraphrase caption for folder: {folder_path}")

def main():
    # Path to the root folder containing the agent_adv directory
    root_dir = "/home/ubuntu/Chen-CA/agent/data/agent_adv"

    from tqdm import tqdm
    
    # Iterate through all folders in the agent_adv directory
    for folder_name in tqdm(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Ensure it is a directory
        if os.path.isdir(folder_path):
            process_caption_file(folder_path)

if __name__ == "__main__":
    main()
