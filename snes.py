from game.game import Game
from trainer.snet import SNET

snet = SNET(Game())
organism = snet.train()

print(organism)
organism.save()
organism.plot_brain()
