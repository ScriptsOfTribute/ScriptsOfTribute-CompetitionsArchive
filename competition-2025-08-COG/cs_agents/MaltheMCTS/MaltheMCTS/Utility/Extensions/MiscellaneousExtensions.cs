using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

namespace MaltheMCTS;

public static class MiscellaneousExtensions{

    /// <summary>
    /// Is not taking Combo effects into account at the moment, but here we should also not look at whether the played card has an effect, but also whether there is a combo cards already played that
    /// will be activated by playing said card
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public static bool IsStochastic(this Move move, SeededGameState gameState)
    {
        if (move.Command == CommandEnum.END_TURN)
        {
            // In this context, i do not count end_turn as stochastic as there is another feature for including chance nodes for end_turn
            return false;
        }
            
        switch(move)
        {
            case SimpleCardMove:
                var simpleCardMove = move as SimpleCardMove;
                switch (simpleCardMove.Command)
                {
                    case CommandEnum.PLAY_CARD:
                    case CommandEnum.ACTIVATE_AGENT:
                        return Utility.RANDOM_EFFECT_CARDS.Contains(simpleCardMove.Card.CommonId);
                    case CommandEnum.ATTACK:
                    case CommandEnum.CALL_PATRON:
                        return false;
                    case CommandEnum.BUY_CARD:
                        return
                            (simpleCardMove.Card.Type == CardType.CONTRACT_ACTION || simpleCardMove.Card.Type == CardType.CONTRACT_AGENT)
                            &&
                            Utility.RANDOM_EFFECT_CARDS.Contains(simpleCardMove.Card.CommonId);
                    case CommandEnum.MAKE_CHOICE:
                        throw new ArgumentException("Simple card move, should not have this command enum: " + CommandEnum.MAKE_CHOICE);
                }
                throw new Exception("Unknown CommandEnum: " + simpleCardMove.Command);
            case SimplePatronMove:
                return false;
            case MakeChoiceMoveUniqueCard:
                // TODO some cards needs to be recognized as causing a stochastic effect on the NEXT move (such as replace tavern effect, where the move afterwards of choosing a card is what causes the random effect)
                // then the last played card needs to be found and if its such an effect (and the choices contains any items), then the move is stochastic. A draft for this exists below
                //var makeChoiceMoveUniqueCard = move as MakeChoiceMoveUniqueCard;
                //var cardChoices = makeChoiceMoveUniqueCard.Choices;
                //var list = gameState.CompletedActions.OrderBy(a => a.TargetCard?.CommonId.ToString() ?? "").ToList();
                //if (gameState.CompletedActions[gameState.CompletedActions.Count - 1].TargetCard?.CommonId == CardId.BARTERER)
                //{
                //}
                return false;
            case MakeChoiceMoveUniqueEffect:
                var makeChoiceMoveUniqueEffect = move as MakeChoiceMoveUniqueEffect;
                var effectChoices = makeChoiceMoveUniqueEffect.Choices;
                return makeChoiceMoveUniqueEffect.Choices.Any(e => e.IsStochastic());
            default:
                throw new Exception("Unknown move type: " + move.GetType().Name);
        }
    }

    public static bool IsStochastic(this UniqueComplexEffect effect) {
        if (effect == null)
        {
            return false;
        }
        // ComplexEffect types:
        // Effect is a single effect
        // EffectComposite is for AND effects
        // EffectOR is for OR effects (choice)
        switch(effect){
                case UniqueEffect:
                    return ((effect as UniqueEffect).Type == EffectType.DRAW);
                case UniqueEffectComposite:
                    // Unfortunately i have to do string comparison here, since the two effect properties are private on EffectComposite
                    return (effect as UniqueEffectComposite).ToSimpleString().Contains("DRAW");
                case UniqueEffectOr:
                    // For OR effects, i say that playing them dont create a random effect, since it only creates the choice.
                    // One of the chosen options can then create a random effect afterwards when chosen, but that is a seperate move
                    return false;
                default:
                    throw new Exception("Unknown effect type: " + effect.GetType().Name);
        }
    }

    public static bool IsInstantPlay(this Move move) {

        if(move.Command == CommandEnum.END_TURN)
        {
            return false;
        }

        switch(move) {
            case SimpleCardMove:
                var simpleCardMove = move as SimpleCardMove;
                switch (simpleCardMove.Command) {
                    case CommandEnum.PLAY_CARD:
                        if (simpleCardMove.Card.Type == CardType.AGENT) {
                            return false;
                        }
                        else if (Utility.INSTANT_EFFECT_PLAY_CARDS.Contains(simpleCardMove.Card.CommonId)){
                            return true;
                        }
                        else {
                            return false;
                        }
                    case CommandEnum.ACTIVATE_AGENT:
                        return Utility.INSTANT_EFFECT_PLAY_CARDS.Contains(simpleCardMove.Card.CommonId);
                    case CommandEnum.BUY_CARD:
                    case CommandEnum.ATTACK:
                        return false;
                    default:
                        throw new Exception("Unexpected simple card move. Command enum: " + simpleCardMove.Command);
                }
            case SimplePatronMove:
            case MakeChoiceMoveUniqueCard:
            case MakeChoiceMoveUniqueEffect:
                return false;
            default:
                throw new Exception("Unknown move type: " + move.GetType().Name);
        }
}
    /// <summary>
    /// Only considered instant play if its an action that only grants resources (gold, power, prestige, patron call, agent health), takes prestige from opponent or forces opponent to discard cards
    /// and does not causes stochasticity or introduces another choice to be made.
    /// </summary>
    public static bool IsInstantPlayEffect(this UniqueComplexEffect effect)
    {
        if (effect == null)
        {
            return true;
        }

        if (effect.IsStochastic())
        {
            return false;
        }

        switch(effect)
        {
            case UniqueEffect:
                var type = (effect as UniqueEffect).Type;
                return type is
                    EffectType.GAIN_COIN or
                    EffectType.GAIN_POWER or
                    EffectType.GAIN_PRESTIGE or
                    EffectType.OPP_LOSE_PRESTIGE or
                    EffectType.PATRON_CALL or
                    EffectType.OPP_DISCARD or
                    EffectType.HEAL; // This is an agent healing itself. Does not introduce a choice
            case UniqueEffectComposite:
                // Unfortunately i have to do string comparison here, since the two effect properties are private on EffectComposite
                return !(new[] { "REPLACE_TAVERN",
                                "AQUIRE_TAVERN",
                                "DESTROY_CARD",
                                "DRAW",
                                "RETURN_TOP",
                                "TOSS",
                                "KNOCKOUT",
                                "CREATE_SUMMERSET_SACKING", // Not sure about this TODO find out
                                }.Any(w => (effect as UniqueEffectComposite).ToSimpleString().Contains(w)));
            case UniqueEffectOr:
                return false;
            default:
                throw new Exception("Unexpected effect type");
        }
    }
}