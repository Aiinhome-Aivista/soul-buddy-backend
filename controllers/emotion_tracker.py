from transformers import pipeline
import collections

class EmotionTracker:
    def __init__(self):
        self.emotion_analyzer = pipeline("text-classification",
                                         model="bhadresh-savani/bert-base-uncased-emotion")
        self.emotion_history = collections.deque(maxlen=20)

    def analyze_emotion(self, text):
        result = self.emotion_analyzer(text)[0]
        emotion = result['label']
        score = result['score']
        self.emotion_history.append((emotion, score))
        return emotion, score

    def get_recent_emotion(self):
        if not self.emotion_history:
            return None
        return self.emotion_history[-1]
