from matplotlib import pyplot as plt

from snes.game.game import Game
from snes.trainer.snet import SNET
# from snake_game.game import Game


snet = SNET(Game())
organism = snet.train()

if organism:
    print(f"Final organism: {organism}")
    organism.save()
    organism.plot_brain()
    plt.show()
