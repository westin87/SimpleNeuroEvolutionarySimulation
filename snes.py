#from game.game import Game
from snake_game.game import Game
from trainer.snet import SNET

snet = SNET(Game())
organism = snet.train()

if organism:
    print(f"Final organism: {organism}")
    organism.save()
    organism.plot_brain()
