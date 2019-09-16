import os
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from torchMoji.examples.botmoji import Botmoji


class ParlAISolver:
    """Dummy class for interacting through transformers with ParlAI
    """
    def __init__(self, config):
        self.botmoji = Botmoji()

        parser = setup_args()
        opt = parser.parse_args(['-mf', os.path.join(config.checkpoint[:-len('1.pkl')] + '/transformer')])
        self.transformer = create_agent(opt, requireModelExists=True)

    def generate_response_to_input(self, raw_text_sentences,
                                   max_conversation_length=5,
                                   max_sentence_length=30, debug=False,
                                   emojize=False, sample_by='priority',
                                   priming_condition=0):

        message = {'id': 'localHuman', 'episode_done': False,
                   'label_candidates': None, 'text': None}

        # Transformer appends message to conversation history stored in its state
        # and outputs reponse
        message['text'] = raw_text_sentences[-1]
        self.transformer.observe(message)
        response = self.transformer.act()['text']
        print(raw_text_sentences[-1])
        if emojize:
            inferred_emojis = self.botmoji.emojize_text(
                raw_text_sentences[-1], 5, 0.07)
            response = inferred_emojis + " " + response

        return response

    def build(self):
        pass
