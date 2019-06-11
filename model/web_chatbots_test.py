import argparse
import os

def interact_with_bot(chatbot, username):
    print("\nPlease enjoy your chat with {0}! Type 'exit' or 'quit' to end the chat at any point.\n".format(chatbot.name))
    messages = []

    while True:

        message = input("[" + username + "]: ")

        if message == "exit" or message == "quit":
            print("Goodbye!")
            return

        if not message:
            continue

        messages.append(message)
        messages = messages[-5:]

        response = chatbot.handle_messages(messages)

        print("[" + chatbot.name + "]: " + response)
        messages.append(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_base_path', type=str, 
                        default='/home/dialog/checkpoints/',
                        help='Base path to checkpoints for model averaging.')
    parser.add_argument('--test_chatbots', action='store_true',
                        help='If true loads bots from test_chatbots instead.')
    parser.set_defaults(test_chatbots=False)
    kwargs = parser.parse_args()

    os.environ['BASE_PATH'] = kwargs.models_base_path
    if kwargs.test_chatbots:
        from model.test_chatbots import chatbots
    else:
        from model.web_chatbots import chatbots

    username = None
    chatbotid = None

    while not username:
        username = input("Please enter your name.\n> ")

    while True:
        while chatbotid not in chatbots.keys():
            chatbotid = input("Please enter the chatbot ID of your choice.\nValid choices are [" + ", ".join(chatbots.keys()) + "]:\n> ")

        chatbot = chatbots[chatbotid]

        interact_with_bot(chatbot, username)

        message = input("Interact with another bot? (y/n): ")

        if message.lower() != 'y':
            break

        chatbotid = None
    