# goal

the purpose of this project is to understand how player populations in a video game respond over time in responses to changes in paramaters for matchmaking. i want to understand and think about how different matchmaking, ie, systems for pairing players up in a competitive game, maintain or reduce the playerbase of a game over time.

# setting

teams are composed of players, where individual players have different levels of skill as proxied for by a skill rating. matchmaking is the system that matches teams together for a competitive game, and the game produces an outcome where some teams _win_ and some teams _lose_. individual players can illustrate skill even in an overall loss by getting _kills_ of the other team.

after a game, each individual's skill rating is updated in response to the results of the game. this then produces feedback that affects future matchmaking.

## matchmaking

for the purpose of creating a matchmaking system, we need to create:

- a system with parameters for measuring and rating individual players. 
- a system with parameters for taking player ratings and converting these to team ratings
- a system for matching teams for the purpose of a match, with parameters for tolerating narrow/wide variance in matchmaking team ratings and matchmaking speed
- a sytem with parameters for updating player ratings in response to the events of a match

### players

for individual players, i want to create parameters such as:

- a parameter for player experience, which we can think of as a proxy for time playing the game, regardless of skill. this is often a player's level in a game that rewards xp and causes level to go up naturally over time.
- a parameter for player skill, which is responsive to individual ability to win matches and gain kills.
- a parameter for player gear, which is a way to measure the value of a gear a player brings into a match. a weight of zero would mean gear has absolutely no impact on player ratings, and therefore no effect on team ratings and matchmaking.

these parameters will then be used additively to create an overall player rating for the purposes of matchmaking.

### teams

- a parameter for aggregating/compressing player ratings into an overall team rating. i envision this as on one end skewing the team rating towards the highest rated player on the team, and on the other end skewing a team's rating towards the lowest skilled player

## search

- a parameter allowing for greater variance in skil/level distributions within matches. with zero tolerance at the one end for a the matchmaking system that only matches within a narrow band, vs extremely high tolerance that allows for wide asymmetries. this should grow as a function of time spent looking for a search

## updates

- a parameter for how much a player skills shift in response to positive and negative outcomes of a match. im envisioning this to be similar to the parameter within an elo system that determines how many points are gained or lost for one match, based on the opponent.

## player base

our player base is composed of a population of individual players that can grow and contract over time. new players can buy the game and join the population, some players stop playing completely, some players might play more sporadically.

at the start of a season, every player is auto assigned the same player rating. over the course of a season, i want to examine how the distribution of player level and player skill ratings shift over time, and how the player base ebs and flows.

to achieve this, i want to design a simulation based approach that is composed of $N$ number of players at the start, with $Growth$ dictating the number of new players that pick up and start playing the game, and $Retention$ indicating how many players stick around over the course of a season.

i then want to simulate seasons given different settings of the parameters i have proposed above and examine how they change and influence a player base over the course of a season.

i want to see how different matchmaking systems (ie, one that does not use player rating at all, and is completely random for matching teams) perform under different settings (highly uniform skill, highly right skewed skill).