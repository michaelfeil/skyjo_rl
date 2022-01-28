import numpy as np
import numpy.ma as ma

num_players = 2
player_won_id = 0
players_cards = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9], list(range(0, 12))])
cards = players_cards.reshape(players_cards.shape[0], -1, 3)
# figure out where we have columns with all same values
cancelout_cols = (np.min(cards, axis=2) == np.max(cards, axis=2))[:, :, np.newaxis]

cancelout_cols = np.broadcast_to(cancelout_cols, (*cancelout_cols.shape[:2], 3))
score = ma.array(cards, mask=cancelout_cols).sum(axis=(1, 2)).compressed()
print(score)

print([0.5, 1, 2] * 2)
