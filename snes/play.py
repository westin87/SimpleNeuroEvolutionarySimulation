from pathlib import Path
from matplotlib import pyplot as plt

from snes.trainer.organism import Organism
from snes.game.game import Game

latest_brain = str(list(Path(__file__).parent.rglob("*.bin"))[-1])

print(latest_brain)
organism = Organism.load(latest_brain, Game)
organism.evaluate()
print(organism.fitness)
organism.plot_brain()
plt.show()
