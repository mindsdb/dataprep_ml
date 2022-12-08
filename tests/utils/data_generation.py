import random
from type_infer.dtype import dtype


test_column_types = {
    'numeric_int': dtype.integer,
    'numeric_float': dtype.float,
    'date_timestamp': dtype.datetime,
    'date_date': dtype.date,
    'categorical_str': dtype.categorical,
    'categorical_int': dtype.categorical,
    'categorical_binary': dtype.binary,
    'sequential_numeric_array': dtype.num_array,
    'sequential_categorical_array': dtype.cat_array,
    'multiple_categories_array_str': dtype.tags,
    'short_text': dtype.short_text,
    'rich_text': dtype.rich_text
}

VOCAB = [
    'remember', 'men', 'pretty', 'break', 'know', 'an', 'forward', 'whose', 'plant', 'decide', 'fit', 'so', 'connect',
    'house', 'then', 'lot', 'protect', 'children', 'above', 'column', 'far', 'continue', 'which', 'discuss', 'test',
    'self', 'dream', 'prepare', 'toward', 'world', 'cold', 'subtract', 'bat', 'subject', 'evening', 'year', 'low',
    'sign', 'very', 'determine', 'sent', 'tube', 'skill', 'first', 'feed', 'fresh', 'note', 'own', 'ball', 'shape',
    'rail', 'drink', 'property', 'did', 'son', 'yet', 'shell', 'believe', 'him', 'noise', 'take', 'spot', 'read',
    'scale', 'but', 'chart', 'wrote', 'war', 'hundred', 'better', 'ease', 'show', 'suggest', 'describe', 'sat', 'camp',
    'oxygen', 'view', 'smile', 'add', 'whole', 'arrange', 'trade', 'money', 'just', 'repeat', 'truck', 'second', 'shoe',
    'wish', 'as', 'mount', 'ground', 'wrong', 'support', 'lead', 'thin', 'agree', 'slave', 'us', 'wide', 'white',
    'room', 'reach', 'are', 'hot', 'thousand', 'science', 'before', 'want', 'broad', 'soon', 'long', 'felt', 'danger',
    'offer', 'saw', 'most', 'rub', 'basic', 'call', 'cost', 'surprise', 'salt', 'office', 'done', 'any', 'egg', 'a',
    'instrument', 'east', 'oil', 'seed', 'state', 'back', 'shop', 'multiply', 'city', 'top', 'arrive', 'ocean',
    'figure', 'good', 'hat', 'claim', 'say', 'he', 'decimal', 'process', 'point', 'hope', 'hunt', 'went', 'night',
    'paragraph', 'hear', 'require', 'near', 'enter', 'collect', 'were', 'region', 'stream', 'teach', 'favor', 'lay',
    'and', 'compare', 'best', 'learn', 'act', 'practice', 'material', 'mean', 'river', 'soldier', 'control', 'equal',
    'between', 'sudden', 'result', 'spend', 'wire', 'wall', 'experiment', 'receive', 'bed', 'condition', 'ship',
    'people', 'your', 'love', 'thank', 'cut', 'who', 'planet', 'drive', 'position', 'select', 'simple', 'temperature',
    'warm', 'power', 'cotton', 'bring', 'them', 'what', 'let', 'kept', 'cross', 'degree', 'center', 'they', 'brother',
    'busy', 'animal', 'wood', 'shall', 'young', 'came', 'is', 'how', 'last', 'able', 'play', 'event', 'village', 'two',
    'many', 'wheel', 'black', 'usual', 'woman', 'will', 'never', 'trouble', 'tell', 'mind', 'dog', 'fig', 'dollar',
    'cover', 'north', 'bought', 'story', 'street', 'suffix', 'laugh', 'instant', 'town', 'rule', 'trip', 'go', 'told',
    'might', 'idea', 'supply', 'mile', 'cow', 'edge', 'rather', 'garden', 'corn', 'parent', 'hour', 'country', 'fast',
    'ten', 'print', 'locate', 'final', 'coast', 'character', 'radio', 'even', 'end', 'stead', 'make', 'snow',
    'possible', 'correct', 'boat', 'here', 'allow', 'month', 'or', 'should', 'same', 'bell', 'matter', 'run', 'beauty',
    'come', 'spread', 'held', 'consonant', 'part', 'food', 'chance', 'sugar', 'history', 'stood', 'out', 'steam',
    'half', 'been', 'draw', 'insect', 'may', 'name', 'music', 'flower', 'through', 'mass', 'map', 'eight', 'man',
    'cent', 'job', 'energy', 'look', 'dad', 'hill', 'million', 'settle', 'song', 'hit', 'does', 'hold', 'pair', 'dress',
    'side', 'cool', 'day', 'gun', 'page', 'until', 'capital', 'appear', 'voice', 'have', 'cause', 'minute', 'wing',
    'keep', 'bone', 'season', 'some', 'also', 'question', 'feel', 'seem', 'necessary', 'these', 'of', 'was', 'against',
    'window', 'don’t', 'chick', 'valley', 'green', 'probable', 'shore', 'fall', 'particular', 'case', 'colony', 'land',
    'place', 'level', 'bear', 'though', 'root', 'weight', 'branch', 'jump', 'true', 'bread', 'yard', 'be', 'element',
    'miss', 'stretch', 'heard', 'lady', 'over', 'present', 'division', 'verb', 'prove', 'ready', 'carry', 'poem',
    'silent', 'poor', 'die', 'death', 'use', 'train', 'anger', 'help', 'substance', 'shine', 'list', 'send', 'syllable',
    'thus', 'brought', 'big', 'now', 'dictionary', 'space', 'unit', 'soil', 'work', 'object', 'board', 'roll', 'six',
    'wonder', 'no', 'sit', 'clock', 'size', 'once', 'front', 'key', 'either', 'if', 'try', 'neighbor', 'our', 'hard',
    'about', 'famous', 'again', 'especially', 'wait', 'think', 'afraid', 'line', 'track', 'quick', 'rose', 'like',
    'field', 'forest', 'numeral', 'path', 'meant', 'color', 'separate', 'copy', 'nation', 'third', 'desert', 'behind',
    'dead', 'spell', 'record', 'teeth', 'lift', 'pattern', 'mountain', 'island', 'soft', 'king', 'since', 'round',
    'made', 'together', 'real', 'floor', 'travel', 'team', 'wife', 'machine', 'plane', 'fish', 'general', 'enough',
    'special', 'natural', 'value', 'join', 'light', 'tie', 'corner', 'rope', 'piece', 'quotient', 'to', 'write',
    'weather', 'old', 'each', 'least', 'provide', 'while', 'log', 'square', 'turn', 'language', 'gas', 'body', 'method',
    'home', 'similar', 'original', 'period', 'circle', 'finish', 'captain', 'fire', 'week', 'post', 'fill', 'count',
    'range', 'well', 'cloud', 'get', 'dark', 'silver', 'occur', 'burn', 'crowd', 'bird', 'double', 'I', 'would', 'this',
    'band', 'quart', 'table', 'rock', 'found', 'friend', 'sight', 'deep', 'dry', 'blood', 'touch', 'fear', 'finger',
    'plan', 'guide', 'hot', 'after', 'hair', 'tree', 'race', 'noon', 'effect', 'wild', 'took', 'hand', 'give', 'clear',
    'noun', 'please', 'do', 'art', 'stay', 'fly', 'whether', 'sell', 'lone', 'from', 'too', 'paint', 'tire', 'loud',
    'divide', 'complete', 'charge', 'left', 'milk', 'spoke', 'base', 'free', 'her', 'human', 'iron', 'choose',
    'continent', 'strange', 'segment', 'summer', 'bit', 'build', 'course', 'type', 'steel', 'press', 'great', 'those',
    'search', 'dear', 'pitch', 'perhaps', 'grand', 'industry', 'quite', 'up', 'term', 'sentence', 'high', 'shout',
    'down', 'we', 'all', 'huge', 'walk', 'solve', 'excite', 'mark', 'pick', 'three', 'other', 'rest', 'law', 'wind',
    'difficult', 'gold', 'populate', 'proper', 'knew', 'one', 'fun', 'seven', 'happen', 'cell', 'throw', 'motion',
    'atom', 'expect', 'live', 'rain', 'stone', 'grass', 'sleep', 'early', 'short', 'always', 'gentle', 'father', 'cook',
    'mouth', 'the', 'inch', 'ago', 'store', 'smell', 'observe', 'magnet', 'leave', 'heart', 'little', 'written',
    'sharp', 'box', 'talk', 'broke', 'score', 'wave', 'bar', 'off', 'century', 'fruit', 'class', 'card', 'way', 'meat',
    'late', 'surface', 'bottom', 'family', 'wash', 'guess', 'catch', 'red', 'fact', 'move', 'visit', 'port', 'set',
    'eat', 'thing', 'when', 'start', 'fair', 'example', 'head', 'four', 'grow', 'earth', 'win', 'gone', 'where',
    'nothing', 'open', 'quiet', 'by', 'can', 'clean', 'group', 'ear', 'moment', 'game', 'close', 'morning', 'reply',
    'straight', 'nature', 'often', 'develop', 'west', 'thick', 'twenty', 'feet', 'led', 'total', 'she', 'slip',
    'create', 'pull', 'system', 'need', 'party', 'sail', 'doctor', 'length', 'has', 'change', 'consider', 'study',
    'rise', 'star', 'operate', 'certain', 'electric', 'leg', 'sheet', 'kind', 'major', 'chair', 'born', 'chord',
    'order', 'ride', 'could', 'word', 'modern', 'face', 'find', 'push', 'me', 'horse', 'differ', 'ever', 'nose',
    'else', 'on', 'spring', 'solution', 'molecule', 'door', 'right', 'enemy', 'symbol', 'paper', 'during', 'sure',
    'tool', 'fight', 'joy', 'stick', 'yes', 'notice', 'station', 'area', 'tall', 'string', 'design', 'tone', 'sky',
    'indicate', 'foot', 'pose', 'success', 'mine', 'air', 'engine', 'listen', 'distant', 'tail', 'invent', 'at']


def generate_short_sentences(n):
    return [' '.join(random.sample(VOCAB, random.randint(2, 6))) for _ in range(n)]


def generate_rich_sentences(n):
    return [' '.join(random.sample(VOCAB, random.randint(7, 16))) for _ in range(n)]
