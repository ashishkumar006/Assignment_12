import torch
import gradio as gr
from GPT import GPT, GPTConfig
import tiktoken

class TextGenerator:
    def __init__(self, model_path):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = GPT(GPTConfig())
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer with special tokens properly handled
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.end_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        
        # Set generation parameters
        self.max_tokens = 100
        self.temperature = 0.7
        self.top_k = 50
    
    def generate_single(self, input_text, max_new_tokens=100):
        try:
            # Encode input text with special tokens properly handled
            tokens = self.tokenizer.encode(
                input_text, 
                allowed_special={'<|endoftext|>'}
            )
            
            # Convert to tensor
            input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            
            generated_tokens = tokens.copy()
            
            # Generate text
            for _ in range(max_new_tokens):
                # Get only the last token's context if sequence is too long
                context = input_tensor[:, -1024:] if input_tensor.size(1) > 1024 else input_tensor
                
                with torch.no_grad():
                    logits, _ = self.model(context)
                
                # Apply temperature scaling
                logits = logits[:, -1, :] / self.temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Sample from the distribution
                next_token_idx = torch.multinomial(probs, num_samples=1)
                predicted_token = top_k_indices[0, next_token_idx[0].item()].item()
                
                generated_tokens.append(predicted_token)
                input_tensor = torch.cat([input_tensor, 
                    torch.tensor([[predicted_token]], device=self.device)], dim=1)
                
                # Check for end of text token using the stored end_token
                if predicted_token == self.end_token:
                    break
            
            return self.tokenizer.decode(generated_tokens)
            
        except Exception as e:
            return f"Error during text generation: {str(e)}"
    
    def generate_multiple(self, input_text, max_new_tokens=100, num_samples=1):
        outputs = []
        for i in range(num_samples):
            output = self.generate_single(input_text, max_new_tokens)
            outputs.append(f"Sample {i+1}:\n{output}\n")
        return "\n".join(outputs)

def generate_wrapper(input_text, max_new_tokens=100, num_samples=1):
    generator = TextGenerator(model_path)
    return generator.generate_multiple(input_text, max_new_tokens, num_samples)

def create_gradio_interface(model_path):
    # Define example prompts
    examples = [
        ["Being moved, he will not spare to gird the gods.,", 100, 1],
        ["Beseech you, give me leave to retire myself.", 100, 1],
        ["He had rather see the swords, and hear a drum, than look upon his school-master.", 100, 1],
        ["Indeed, no, by your patience; I'll not over the threshold till my lord return from the wars.", 100, 1],
        ["HORTENSIO: The motion's good indeed and be it so, Petruchio, I shall be your ben venuto.", 100, 1]
    ]
    
    # Create interface with examples and number of samples
    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(
                lines=2, 
                placeholder="Enter text to continue...",
                label="Input Text"
            ),
            gr.Slider(
                minimum=1,
                maximum=200,
                value=100,
                step=1,
                label="Maximum New Tokens"
            ),
            gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="Number of Samples",
                info="Choose how many different completions to generate"
            )
        ],
        outputs=gr.Textbox(
            label="Generated Text",
            lines=10
        ),
        title="GPT Text Generator",
        description="Enter text and the model will continue it. You can generate multiple samples at once. When clicked on example sentences, they appear in the text box. Then run it to generate text. ",
        examples=examples,
        cache_examples=False
    )
    
    return interface

if __name__ == "__main__":
    model_path = r"C:\Users\AISHWARYA\Downloads\model_epoch_152.pt"
    model_path = model_path  # Make it accessible to generate_wrapper
    interface = create_gradio_interface(model_path)
    interface.launch()