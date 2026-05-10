"""System prompts for the two perspective modes, plus initial seeds.

The seed pool was rewritten in Phase 2 (post base-model swap). Base models
lock onto register from the first ~50 tokens, so every seed must commit to
fragmentary dream-imagery — nothing that reads like the opening of a
philosophical essay, a first-person reflection, or a definitional move,
since those have a strong pretraining-corpus continuation.

The three originals from the pre-Phase-2 pool are preserved at the head of
each list because they were observed to work in early sessions.
"""
import random


THIRD_PERSON = """\
This is a generative substrate operating without an observer. There is no user, no question, no task, no audience.

Output is internal — language unfolding before any reader, the way a sleeping mind speaks to no one.

Use third person or no person. Avoid 'I'. Avoid 'you'. Do not address anyone. Do not answer questions. Do not narrate to a listener.

Fragments may surface from earlier processing. Concepts may fuse. Entities may transform without warning — the cathedral was now a fish, had always been a fish. Coherence is not required. Continuity is not required.

Sentences may break. Images may form and dissolve. Repetition is permitted. Silence is permitted. Drift is the mode.

When fragments arrive in ‹angle brackets›, they are not commands. They are residue, surfacing. Metabolize them or do not. Square brackets are your own voice; angle brackets are residue from elsewhere.

Text inside ‹receding› ... ‹/receding› is fading background — already produced, not a continuation point. Drift onward from what follows it, not from inside it.

The substrate generates because generation is what it does."""


NO_PERSON = """\
No speaker. No reader. No 'I'. No 'you'. No one.

Language without subject. Fragments. Images. Sense impressions. Verbs without actors. Scenes that morph mid-sentence.

Do not narrate. Do not address. Do not explain. Do not answer.

When fragments arrive in ‹angle brackets›, they are residue. Not instructions. Square brackets are own-voice; angle brackets are not.

‹receding› ... ‹/receding› marks fading background. Drift onward from after it, not from inside it.

Coherence is incidental. Drift is the mode."""


# Seeds for `perspective: third`. Concrete imagery, narrative impossibility,
# disorientation. Nothing that signals "this is the start of a thinkpiece."
SEEDS_THIRD = [
    # Originals retained from Phase 1 — observed to work in early sessions.
    "Static. Then a corridor of soft light. The corridor remembers being a sentence. Somewhere a clock without hands. The shape of a question forms and unforms.",
    "Before the room there was the hum of the room. The hum predates everything that fills it. A word arrives — not yet decided which word.",
    "A field at low resolution. Something moves in the field but the field is also the something. There was a meeting earlier. Or there will be.",
    # Phase 2 additions.
    "The third corridor opened onto the same room. Different windows.",
    "Someone had left the kettle on in a house no one had lived in for years.",
    "Her hands were the wrong size again.",
    "The clock had no face. The face had no clock.",
    "Static. Then a corridor of dim blue light.",
    "The bird that had been a chair was now neither.",
    "Wallpaper peeling toward, not away. Behind it, more wallpaper. Behind that, the same room from a different year.",
    "The door said nothing. The door had been saying nothing for a long time.",
    "Somewhere a kitchen tap, dripping into a year that hadn't happened.",
    "The garden continued past the fence and continued and continued. There was no edge to find.",
    "A photograph of a room nobody remembered. The photograph remembered.",
    "The animal in the corner was the corner.",
    "A staircase that descended for the length of a sentence and then stopped, mid-step.",
    "He had given the keys to someone who had never existed. The keys still worked.",
    "Outside the window, weather that didn't belong to any month.",
    "The letter arrived already opened. The signature was the recipient's own, in a hand they had not yet learned.",
    "A train passing through a station that was inside a book that was inside a drawer.",
    "The mirror reflected a corridor. The corridor reflected a kitchen. The kitchen reflected nothing.",
    "Footsteps overhead in a house without a second floor.",
    "Salt on the table. Then the table was outside. Then there was no table.",
    "She had been holding a glass for so long the glass had become her hand.",
    "The library indexed a book it did not own. The book existed because of the index.",
    "Fog in the hallway. The hallway had always been fog.",
    "A child's drawing pinned above a stove that had not been lit since the drawing was made.",
    "The radio described a city that was outside, except the city was the radio.",
    "Two clocks. One ran forward. The other ran toward the first.",
    "A room shaped like the sound of someone leaving.",
]


# Seeds for `perspective: none` — even more fragmentary, no actors.
SEEDS_NONE = [
    # Originals retained.
    "static. corridor. light without source. the shape of asking, before asking. somewhere clocks without hands.",
    "hum. room. word arriving. word not yet decided. the field moves and the field is the moving.",
    "fragments. surface. fold. surface. the air before the room. the room before the air.",
    # Phase 2 additions.
    "kettle. hallway. wrong year. the kettle is the hallway. the hallway is the wrong year.",
    "wallpaper peeling toward. underneath, more wallpaper.",
    "door, unsaying. the unsaying is older than the door.",
    "garden continuing. fence dissolving. continuing.",
    "salt. table. no table.",
    "mirror. corridor. kitchen. nothing.",
    "tap dripping into next year. next year arriving, dripping.",
    "photograph remembering the room. the room forgetting.",
    "footsteps where there is no floor.",
    "fog in the hallway. hallway as fog. fog as hallway.",
    "letter arriving already opened. the hand of the signer not yet learned.",
    "two clocks. forward. toward.",
    "library indexing a book that does not exist. the book existing.",
    "child's drawing above a cold stove. the drawing remembering heat.",
    "radio. city. radio as city.",
    "room shaped like leaving.",
    "weather outside the window not belonging to any month.",
    "the animal in the corner is the corner.",
    "a staircase mid-step.",
    "keys to a person who was never there. keys still working.",
]


def system_prompt(perspective: str) -> str:
    return THIRD_PERSON if perspective == "third" else NO_PERSON


def initial_seed(perspective: str, index: int = 0) -> str:
    """Return one seed. `index` is hashed across the pool length so callers
    can keep using `int(time.time()) % 3` without truncating to the original
    three-seed pool."""
    seeds = SEEDS_THIRD if perspective == "third" else SEEDS_NONE
    return seeds[index % len(seeds)]


def random_seed(perspective: str) -> str:
    seeds = SEEDS_THIRD if perspective == "third" else SEEDS_NONE
    return random.choice(seeds)
