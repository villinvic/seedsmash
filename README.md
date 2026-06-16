# SeedSmash

## next todos:
- fully delayed bots
- shared core module + split heads
- train shared core with gradients / sqrt(n)
- have an oracle critic ??? either compute value on server, but must bootstrap with 0, or compute on workers but will be slow
  - I think not
- use fictitious play (maybe more often for bottom bots)
  - lowest the bot, more likely it will play again a fixed version
- spend time on rewards
  - do a test without pseudo reward, just kills/damage
  - three bots, puff, dk, m2 ?
- some helper functions are not "helper functions", should changed as is.
- implement missing characters
  - zelda/sheik
  - ice climbers
  - kirby
- update website
  - online/offline indicator
  - running session name, time

## implement missing characters
-> zelda/sheik
-> ice climbers
-> kirby

-> test with 3 chars: DK, samus, g&w ?
-> reclean rewards
--> id



-> share core to all bots
--> how do we handle delay ? everyone has same delay ?
-> split heads, heads have char specific input fed there


-> fictitious play ?
--> a bot that falls behind just learns to flee
--> making it optimistic does not solve the issue/ is hard to use: it will just be biased
--> fictitious play can solve than storing a snapshot of every bot every now and then
--> interaction graph would be:
---> sample a bot, starting from bottom rank: cascading sampling ?
---> sample opponent from snapshots:
	- uniformly ?
	- using elo ?
	- using elo slightly stronger ?
	-- opponent stronger -> robustness
	-- opponent weaker -> exploits 
	-- I think stronger is better
	--> offset elo a bit higher and sample opponent
	--> allow self-play
	-- Opponent Pool as a bot configuration!

-> how to handle new bots ?
--> neglect for now


--> request matchup on stream: has delays
--> cancel current game.
--> does it work as intended ?

## pbrl ?

-> each bot would have their own reward function (in policyparams)
-> starting with a +1 - 1 for kills, +-0.005 for damage
--> hard coded in the game

-> every now and then:
	- generate a sequence of clips to compare
	--> record sequence of state embeddings, and timestamps for the replay
	- upload to website, erase old ones, maybe a queue
	- prompt user
	- when user submits preferences, system updates reward function
	- 