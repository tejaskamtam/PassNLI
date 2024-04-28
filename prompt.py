from env import OPENAI_KEY, ANTHROPIC_KEY
import argparse
import openai, anthropic
import os, sys, json

class HiddenPrints:
    def __init__(self, mute=False):
        self.mute = mute
    def __enter__(self):
        if self.mute:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mute:
            sys.stdout.close()
            sys.stdout = self._original_stdout




def parse_args():
    parser = argparse.ArgumentParser(description='Prompt ChatGPT')

    parser.add_argument('--model', '-m', type=str, default="gpt", choices=['gpt', 'claude'], help="Model to use.")
    parser.add_argument('--version', '-v', type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'haiku', 'sonnet', 'opus'], help="Version of the model to use.")
    parser.add_argument('--save-history', '-s', nargs='?', default=None, const='history.jsonl', help="APPEND chat completion to a file. Default to 'history.jsonl'.")
    parser.add_argument('--load-history', '-l', nargs='?',  default=None, const='history.jsonl', help='.jsonl file to load history as json objects from')
    parser.add_argument('--temperature', '-t', type=float, default=1, help='Sampling temperature (default 1)')
    parser.add_argument('--top-k', '-k', type=int, default=1, help='Top-k sampling.')
    parser.add_argument('--prompt', '-p', required=True, type=str, help='Prompt for the chat')
    parser.add_argument('--system-prompt', '-sys', type=str, default=None, help='System prompt for the chat')
    parser.add_argument('--max-tokens', '-max', type=int, default=150, help='Maximum tokens to generate')
    parser.add_argument('--mute', action='store_true', help='Mute the output')

    args = parser.parse_args()
  
    if args.model == 'claude' and args.version == 'gpt-3.5-turbo':
        args.version = 'haiku'
    if args.version == 'haiku':
        args.version = 'claude-3-haiku-20240307'
    if args.version == 'sonnet':  
        args.version = 'claude-3-sonnet-20240229'
    if args.version == 'opus':   
        args.version = 'claude-3-opus-20240229'
    return args


# Create a generic function that calls the OpenAI API for chat completion
def chat(args, client, content, history=None):
    # print("CONTENT:",content)
    if args.model == 'gpt':
        messages = [{ 'role': 'system', 'content': args.system_prompt }] if args.system_prompt else []
        messages += history if history else []
        messages.append({ 'role': 'user', 'content': content })
        # print("MESSAGES:","%r" % messages)
        completion = client.chat.completions.create(
            model=args.version,
            temperature=args.temperature,
            n=args.top_k,
            messages=messages,
        )
        # Convert completions response to a dictionary
        # print("RAW COMPLETION:",completion)
        return completion

    if args.model == 'claude':
        messages = history if history else []
        messages.append({ 'role': 'user', 'content': content })
        completion = client.messages.create(
            model=args.version,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            system=args.system_prompt if args.system_prompt else "default",
            messages=messages,
        )
        ### NOTE: only returns top_1 completion atm
        return completion
    return None

if __name__ == '__main__':
    args = parse_args()

    if args.model == 'gpt':
        client = openai.OpenAI(api_key=OPENAI_KEY)
    if args.model == 'claude':
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    history = None

    if args.load_history:
        with open(args.load_history, 'r') as f:
            # read raw lines without escaping
            history = [json.loads(line) for line in f]
    # print("PROMPT: ", args.prompt)

    ## RAW COMPLETION(s)
    completions = chat(args, client, args.prompt, history)

    with HiddenPrints(args.mute):
        print("RAW COMPLETION:",completions)
    
    if args.model == 'gpt':
        with HiddenPrints(args.mute):
            for k in range(args.top_k):
                print("ASSSISTANT:", completions.choices[k].model_dump()["message"]["content"], "\n")
    elif args.model == 'claude':
        with HiddenPrints(args.mute):
            print("ASSSISTANT:", completions.content[0].text, "\n")
    else:
        raise ValueError("Invalid model type. Choose 'gpt' or 'claude'.")

    if args.save_history:
        with open(args.save_history, 'a+') as f:
            f.write(json.dumps({'role':'user', 'content':str(args.prompt)}) + '\n')
            if args.model == 'gpt':
                f.write(json.dumps({'role':'assistant', 'content':str(completions.choices[0].model_dump()["message"]["content"])}) + '\n')
            elif args.model == 'claude':
                f.write(json.dumps({'role':'assistant', 'content':completions.content[0].text}) + '\n')
            else:
                raise ValueError("Invalid model type. Choose 'gpt' or 'claude'.")
    sys.exit(0)