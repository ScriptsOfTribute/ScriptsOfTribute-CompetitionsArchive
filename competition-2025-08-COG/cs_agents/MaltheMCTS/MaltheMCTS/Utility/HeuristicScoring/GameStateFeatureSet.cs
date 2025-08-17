using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute;
using ScriptsOfTribute.Serializers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.HeuristicScoring;

namespace SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring
{
    /// <summary>
    /// Although ML.Net only wants floats, i keep this version as its more clean and accurate, and is ready to be used by a better framework. Instead a transformation to
    /// an object with float properties is done before working with model through ML.Net
    /// </summary>
    public struct GameStateFeatureSet
    {
        public int CurrentPlayerPrestige { get; set; }
        public double CurrentPlayerDeck_PrestigeStrength { get; set; }
        public double CurrentPlayerDeck_PowerStrength { get; set; }
        public double CurrentPlayerDeck_GoldStrength { get; set; }
        public double CurrentPlayerDeck_MiscStrength { get; set; }
        public double CurrentPlayerDeckComboProportion { get; set; }
        public double CurrentPlayerAgent_PrestigeStrength { get; set; }
        public double CurrentPlayerAgent_PowerStrength { get; set; }
        public double CurrentPlayerAgent_GoldStrength { get; set; }
        public double CurrentPlayerAgent_MiscStrength { get; set; }
        public int CurrentPlayerPatronFavour { get; set; }
        public int OpponentPrestige { get; set; }
        public double OpponentDeck_PrestigeStrength { get; set; }
        public double OpponentDeck_PowerStrength { get; set; }
        public double OpponentDeck_GoldStrength { get; set; }
        public double OpponentDeck_MiscStrength { get; set; }
        public double OpponentAgent_PrestigeStrength { get; set; }
        public double OpponentAgent_PowerStrength { get; set; }
        public double OpponentAgent_GoldStrength { get; set; }
        public double OpponentAgent_MiscStrength { get; set; }
        public int OpponentPatronFavour { get; set; }
        public double? WinProbability { get; set; }

        public GameStateFeatureSetCsvRow ToCsvRow()
        {
            return new GameStateFeatureSetCsvRow
            {
                CurrentPlayerPrestige = (float)CurrentPlayerPrestige,
                CurrentPlayerDeck_PrestigeStrength = (float)CurrentPlayerDeck_PrestigeStrength,
                CurrentPlayerDeck_PowerStrength = (float)CurrentPlayerDeck_PowerStrength,
                CurrentPlayerDeck_GoldStrength = (float)CurrentPlayerDeck_GoldStrength,
                CurrentPlayerDeck_MiscStrength = (float)CurrentPlayerDeck_MiscStrength,
                CurrentPlayerDeckComboProportion = (float)CurrentPlayerDeckComboProportion,
                CurrentPlayerAgent_PrestigeStrength = (float)CurrentPlayerAgent_PrestigeStrength,
                CurrentPlayerAgent_PowerStrength = (float)CurrentPlayerAgent_PowerStrength,
                CurrentPlayerAgent_GoldStrength = (float)CurrentPlayerAgent_GoldStrength,
                CurrentPlayerAgent_MiscStrength = (float)CurrentPlayerAgent_MiscStrength,
                CurrentPlayerPatronFavour = (float)CurrentPlayerPatronFavour,
                OpponentPrestige = (float)OpponentPrestige,
                OpponentDeck_PrestigeStrength = (float)OpponentDeck_PrestigeStrength,
                OpponentDeck_PowerStrength = (float)OpponentDeck_PowerStrength,
                OpponentDeck_GoldStrength = (float)OpponentDeck_GoldStrength,
                OpponentDeck_MiscStrength = (float)OpponentDeck_MiscStrength,
                OpponentAgent_PrestigeStrength = (float)OpponentAgent_PrestigeStrength,
                OpponentAgent_PowerStrength = (float)OpponentAgent_PowerStrength,
                OpponentAgent_GoldStrength = (float)OpponentAgent_GoldStrength,
                OpponentAgent_MiscStrength = (float)OpponentAgent_MiscStrength,
                OpponentPatronFavour = (float)OpponentPatronFavour
            };
        }
    }

    public static class FeatureSetUtility
    {
        private const double AGENT_HP_MULTIPLIER = 0.15;
        private const double CHOICE_WEIGHT = 0.75;

        public static GameStateFeatureSet BuildFeatureSet(SeededGameState gameState)
        {
            // base resources
            int currentPlayerPrestige = gameState.CurrentPlayer.Prestige;
            if (gameState.EnemyPlayer.Agents.All(a => !a.RepresentingCard.Taunt))
            {
                //this simplification is becuase this feature set represents end of turn
                currentPlayerPrestige += gameState.CurrentPlayer.Power;
            }
            int opponentPrestige = gameState.EnemyPlayer.Prestige;
            // decks (and combo synergies)
            var currentPlayerCompleteDeck = new List<Card>();
            currentPlayerCompleteDeck.AddRange(gameState.CurrentPlayer.Hand);
            currentPlayerCompleteDeck.AddRange(gameState.CurrentPlayer.DrawPile);
            currentPlayerCompleteDeck.AddRange(gameState.CurrentPlayer.Played);
            currentPlayerCompleteDeck.AddRange(gameState.CurrentPlayer.CooldownPile);
            currentPlayerCompleteDeck.AddRange(gameState.CurrentPlayer.Agents.Where(a => a.RepresentingCard.Type != CardType.CONTRACT_AGENT).Select(a => a.RepresentingCard));
            var currentPlayerPatronToDeckRatio = GetPatronRatios(currentPlayerCompleteDeck, gameState.Patrons);
            var currentPlayerDeckStrengths = ScoreStrengthsInDeck(currentPlayerCompleteDeck, currentPlayerPatronToDeckRatio);

            // To allow model to value putting combo cards into your deck before they have an effect
            double currentPlayerDeckComboProportion = ((double)currentPlayerCompleteDeck.Where(c => c.Deck != PatronId.TREASURY).Count()) / currentPlayerCompleteDeck.Count;

            var opponentCompleteDeck = new List<Card>();
            opponentCompleteDeck.AddRange(gameState.EnemyPlayer.DrawPile);
            opponentCompleteDeck.AddRange(gameState.EnemyPlayer.Played);
            opponentCompleteDeck.AddRange(gameState.EnemyPlayer.CooldownPile);
            opponentCompleteDeck.AddRange(gameState.EnemyPlayer.Agents.Where(a => a.RepresentingCard.Type != CardType.CONTRACT_AGENT).Select(a => a.RepresentingCard));
            var opponentPatronToDeckRatio = GetPatronRatios(opponentCompleteDeck, gameState.Patrons);
            var opponentDeckStrengths = ScoreStrengthsInDeck(opponentCompleteDeck, opponentPatronToDeckRatio);

            // agents
            var currentPlayerAgentStrengths = ScoreStrengthsInDeck(gameState.CurrentPlayer.Agents, currentPlayerPatronToDeckRatio, currentPlayerCompleteDeck.Count);
            var opponentAgentStrengths = ScoreStrengthsInDeck(gameState.EnemyPlayer.Agents, opponentPatronToDeckRatio, opponentCompleteDeck.Count);

            // patrons. Maybe needs to be more sophisticated. Right now, does not look of benefit of having specific patron favours, but just the amount 
            int currentPlayerPatronFavour = 0;
            int opponentPatronFavour = 0;

            foreach (var patron in gameState.Patrons)
            {
                var favouredPlayer = gameState.PatronStates.GetFor(patron);

                if (favouredPlayer == gameState.CurrentPlayer.PlayerID)
                {
                    currentPlayerPatronFavour++;
                }
                else if (favouredPlayer == gameState.EnemyPlayer.PlayerID)
                {
                    opponentPatronFavour++;
                }
            }

            var featureSet = new GameStateFeatureSet()
            {
                CurrentPlayerPrestige = currentPlayerPrestige,
                CurrentPlayerDeck_PrestigeStrength = currentPlayerDeckStrengths.PrestigeStrength,
                CurrentPlayerDeck_PowerStrength = currentPlayerDeckStrengths.PowerStrength,
                CurrentPlayerDeck_GoldStrength = currentPlayerDeckStrengths.GoldStrength,
                CurrentPlayerDeck_MiscStrength = currentPlayerDeckStrengths.MiscellaneousStrength,
                CurrentPlayerDeckComboProportion = currentPlayerDeckComboProportion,
                CurrentPlayerAgent_PrestigeStrength = currentPlayerAgentStrengths.PrestigeStrength,
                CurrentPlayerAgent_PowerStrength = currentPlayerAgentStrengths.PowerStrength,
                CurrentPlayerAgent_GoldStrength = currentPlayerAgentStrengths.GoldStrength,
                CurrentPlayerAgent_MiscStrength = currentPlayerAgentStrengths.MiscellaneousStrength,
                CurrentPlayerPatronFavour = currentPlayerPatronFavour,
                OpponentPrestige = opponentPrestige,
                OpponentDeck_PrestigeStrength = opponentDeckStrengths.PrestigeStrength,
                OpponentDeck_PowerStrength = opponentDeckStrengths.PowerStrength,
                OpponentDeck_GoldStrength = opponentDeckStrengths.GoldStrength,
                OpponentDeck_MiscStrength = opponentDeckStrengths.MiscellaneousStrength,
                OpponentAgent_PrestigeStrength = opponentAgentStrengths.PrestigeStrength,
                OpponentAgent_PowerStrength = opponentAgentStrengths.PowerStrength,
                OpponentAgent_GoldStrength = opponentAgentStrengths.GoldStrength,
                OpponentAgent_MiscStrength = opponentAgentStrengths.MiscellaneousStrength,
                OpponentPatronFavour = opponentPatronFavour,
            };

            return featureSet;
        }

        public static Dictionary<PatronId, double> GetPatronRatios(List<Card> deck, List<PatronId> patrons)
        {
            var patronToAmount = new Dictionary<PatronId, int>();
            var patronToDeckRatio = new Dictionary<PatronId, double>();

            foreach (var patron in patrons)
            {
                patronToAmount.Add(patron, 0);
            }

            foreach (var currCard in deck)
            {
                patronToAmount[currCard.Deck]+=1;
            }

            foreach (var currPair in patronToAmount)
            {
                patronToDeckRatio.Add(currPair.Key, (double)currPair.Value / deck.Count);
            }

            return patronToDeckRatio;
        }

        private static CardStrengths ScoreStrengthsInDeck(List<SerializedAgent> agents, Dictionary<PatronId, double> patronToDeckRatio, int deckSize)
        {
            var result = new CardStrengths();

            foreach (var agent in agents)
            {
                var agentCardStrength = ScoreStrengthsInDeck(agent.RepresentingCard, patronToDeckRatio[agent.RepresentingCard.Deck], deckSize);
                var agentStrength = agentCardStrength + (agentCardStrength * AGENT_HP_MULTIPLIER * agent.CurrentHp); // A way of contributing extra strengths to the agent, the more HP it has
                if (agent.RepresentingCard.Taunt)
                {
                    agentStrength.PrestigeStrength += agent.CurrentHp;
                }

                result += agentStrength;
            }

            return result;
        }

        private static CardStrengths ScoreStrengthsInDeck(List<Card> deck, Dictionary<PatronId, double> patronToDeckRatio)
        {
            var summedStrengths = new CardStrengths();

            foreach (var currCard in deck)
            {
                summedStrengths += ScoreStrengthsInDeck(currCard, patronToDeckRatio[currCard.Deck], deck.Count);
            }

            // FUTURE maybe this is where we need to look at draw effects afterwards

            return summedStrengths / deck.Count;
        }

        public static CardStrengths ScoreStrengthsInDeck(Card card, double patronToDeckRatio, int deckSize)
        {
            var result = new CardStrengths();
            foreach (var effect in card.Effects)
            {
                if (effect == null)
                {
                    continue;
                }
                else
                {
                    var uniqueEffect = effect.MakeUniqueCopy(card.CreateUniqueCopy()); // TODO refactor to simply use left and right on effect if it gets readable
                    var cardStrength = ScoreComplexEffectStrengthsInDeck(uniqueEffect, patronToDeckRatio, deckSize);
                    if (card.Type == CardType.AGENT || card.Type == CardType.CONTRACT_AGENT) {
                        cardStrength += cardStrength * (AGENT_HP_MULTIPLIER * card.HP);
                        if (card.Taunt)
                        {
                            cardStrength.PrestigeStrength += card.HP;
                        }
                    }
                    result += cardStrength;
                }
            }

            return result;
        }

        /// <summary>
        /// TODO refactor to simply use ComplexEffect if left and right are made readable there
        /// </summary>
        private static CardStrengths ScoreComplexEffectStrengthsInDeck(UniqueComplexEffect effect, double patronToDeckRatio, int deckSize)
        {
            switch (effect)
            {
                case UniqueEffect:
                    return ScoreEffectStrengthsInDeck((effect as UniqueEffect)!, patronToDeckRatio, deckSize);
                case UniqueEffectComposite:
                    var effectComposite = (effect as UniqueEffectComposite)!;
                    var effect1Strengths = ScoreEffectStrengthsInDeck(effectComposite.GetRight(), patronToDeckRatio, deckSize);
                    var effect2Strengths = ScoreEffectStrengthsInDeck(effectComposite.GetLeft(), patronToDeckRatio, deckSize);
                    return effect1Strengths + effect2Strengths;
                case UniqueEffectOr:
                    var effectOr = (effect as UniqueEffectOr)!;
                    var effectaStrengths = ScoreEffectStrengthsInDeck(effectOr.GetRight(), patronToDeckRatio, deckSize);
                    var effectbStrengths = ScoreEffectStrengthsInDeck(effectOr.GetLeft(), patronToDeckRatio, deckSize);
                    // A way to give reward for both choices, but give a penalty for not being able to apply both
                    return effectaStrengths * CHOICE_WEIGHT + effectbStrengths * CHOICE_WEIGHT;
                default:
                    throw new ArgumentException("Unexpected effect type: " + effect.GetType().Name);
            }
        }

        private static CardStrengths ScoreEffectStrengthsInDeck(Effect effect, double patronToDeckRatio, int deckSize)
        {
            var result = new CardStrengths();
            switch (effect.Type)
            {
                case EffectType.ACQUIRE_TAVERN:
                case EffectType.CREATE_SUMMERSET_SACKING:
                case EffectType.DESTROY_CARD:
                // FUTURE use overall strengths of deck if possible
                case EffectType.DRAW:
                case EffectType.HEAL:
                case EffectType.OPP_DISCARD:
                case EffectType.PATRON_CALL:
                case EffectType.REPLACE_TAVERN:
                case EffectType.RETURN_TOP:
                case EffectType.TOSS:
                case EffectType.KNOCKOUT_ALL:
                case EffectType.RETURN_AGENT_TOP:
                    // FUTURE Do something more sophisticated with these
                    result.MiscellaneousStrength += 1;
                    break;
                case EffectType.KNOCKOUT:
                case EffectType.DONATE:
                    // TODO fix hardcoded value.
                    // weight because its not as good as a draw since you have to discard as well
                    result.MiscellaneousStrength += (effect.Amount * 0.75);
                    break;
                case EffectType.GAIN_COIN:
                    result.GoldStrength += effect.Amount;
                    break;
                case EffectType.GAIN_POWER:
                    result.PowerStrength += effect.Amount;
                    break;
                case EffectType.GAIN_PRESTIGE:
                case EffectType.OPP_LOSE_PRESTIGE:
                    result.PrestigeStrength += effect.Amount;
                    break;
            }

            if (effect.Combo > 1)
            {
                result = result * GetComboProbability(effect, patronToDeckRatio, deckSize);
            }

            return result;
        }

        private static double GetComboProbability(Effect effect, double patronToDeckRatio, int deckSize)
        {
            // FUTURE replace with bionomial calculation as this is inaccurate as every time you draw a card beside this patron, the probability of drawing this patron is increased and vice versa (since you cant draw the same cards multiple times)
            double drawProbability = 5 * patronToDeckRatio; //We draw 5 cards at start of each turn
            return Math.Pow(drawProbability, effect.Combo);
        }
    }

}
