# Tales of Tribute AI Competition at IEEE Conference on Games 2024

Tales of Tribute AI Competition has been one of the events at [IEEE COG 2024](https://2024.ieee-cog.org/competitions).


The Scripts of Tribute version was [1.1](https://github.com/ScriptsOfTribute/ScriptsOfTribute-GUI/releases/tag/v1.1) featuring 7 patron decks (Pelin, Hlaalu, Crows, Ansei, Rajhin, Red Eagle, Orgnum) and  compatible with Tales of Tribute from ESO PC/Mac Patch 9.2.10 (25.02.2024).

### Results


![Winners](./winners.png)

The results of the competition were officially presented during the conference: [presentation slides](./slides.pdf).


The winnng agent, developed by Adam Ciężkowski Artur Krzyżyński is described in their [engineer's thesis](https://jakubkowalski.tech/Supervising/Ciezkowski2023DevelopingCard.pdf).


![Results chart](./graph.svg)



### Reproducibility

Here is the code allowing to rerun the competition.

All participating agents are available in the repository.

```sh
# Stock Ubuntu 24.04
source setup.sh
source prepare.sh
source run.sh | tee out-1.txt & # One per server.
source run.sh | tee out-2.txt & # One per server.
source run.sh | tee out-3.txt & # One per server.
source run.sh | tee out-4.txt & # One per server.
wait
source graph.sh
```


## Archival Call for Participants


#### Changes from 2023 edition

- Added Orgnum deck
- Applied balance changes compatible with latest ESO patch
- Added an adapter to allow languages other than C# (more information how to use the adapter [here](https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core?tab=readme-ov-file#external-language-adapter-docs))
- Added a [Dockerfile](https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core/blob/master/Dockerfile)
- Multiple QoL changes for writing and testing agents


### Important Dates

- **22nd July 2024**, 23:59 GMT - **Submission deadline**
- 5th-8th August 2024 - [COG conference](https://2024.ieee-cog.org/) and results announcement


### Submission Rules

- Please send a single `.cs` file containing your agent's source code or a zip archive with all the others necessary files to jko@cs.uni.wroc.pl.
- In case of agents written in other programming languages please attach compilation/run instructions.
- Additionally, the email should contain:
  - Agent's name.
  - Names (and institutions, if any) of all agent's authors.
  - Short description of the agent. Preferably a few slides or a short note in markdown or PDF; it has to describe what does the agent do, e.g., whether it employs some search algorithms or neural networks.
- Multiple bots can be submitted, but please indicate if a submission should replace an old one or be counted as a new submission (with a different agent's name). Each participant can have up to 2 final submissions. 
- Please be aware that submitted agents are going to be published in this repository after the competition. With the submission, you agree with this procedure.


### Evaluation

Agents will be evaluated using the [SoT-Core Game Runner](https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core), on a large number of mirror matches using randomly generated seeds in an all-play-all system. The deciding factor will be the average winrate.

Evaluation environment will be compatible with the one provided by [Dockerfile](https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core/blob/master/Dockerfile).

Time limit:
- 10 seconds for every turn

Memory limit and other constraints:
- while playing, the bot should not exceed 256 MB of memory. Anytime exceedance of 1024 MB of RAM usage will result in excluding the bot from the contest
- the size of sent file/archive should not exceed 25 MB


Game version:
- compatible with Tales of Tribute from ESO PC/Mac Patch 9.2.10 (25.02.2024)
- 7 patrons available: [Pelin](https://en.uesp.net/wiki/Online:Saint_Pelin), [Hlaalu](https://en.uesp.net/wiki/Online:Grandmaster_Delmene_Hlaalu), [Crows](https://en.uesp.net/wiki/Online:Duke_of_Crows_(Patron)), [Ansei](https://en.uesp.net/wiki/Online:Ansei_Frandar_Hunding), [Rajhin](https://en.uesp.net/wiki/Online:Rajhin), [Red Eagle](https://en.uesp.net/wiki/Online:Red_Eagle), and [Orgnum](https://en.uesp.net/wiki/Online:Sorcerer-King_Orgnum).
- all decks are assumed to be fully upgraded



### Prizes

- $500USD for the first place
- $300USD for the second place
- $200USD for the third place

Prize founded by the [IEEE CIS Education Competition Subcommittee](https://cis.ieee.org/).

