using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

namespace MaltheMCTS;

public static class IsIdenticalExtensions{

    public static bool IsIdentical(this SeededGameState instance, SeededGameState other) {

        return ( 
                instance.ComboStates.IsIdentical(other.ComboStates)
                // Current player
            &&  instance.CurrentPlayer.Agents.IsIdentical(other.CurrentPlayer.Agents)
            &&  instance.CurrentPlayer.Coins == other.CurrentPlayer.Coins
            &&  instance.CurrentPlayer.CooldownPile.IsIdentical(other.CurrentPlayer.CooldownPile)
            &&  instance.CurrentPlayer.DrawPile.IsIdentical(other.CurrentPlayer.DrawPile)
            &&  instance.CurrentPlayer.Hand.IsIdentical(other.CurrentPlayer.Hand)
            &&  instance.CurrentPlayer.KnownUpcomingDraws.IsIdentical(other.CurrentPlayer.KnownUpcomingDraws)
            &&  instance.CurrentPlayer.PatronCalls == other.CurrentPlayer.PatronCalls
            &&  instance.CurrentPlayer.Played.IsIdentical(other.CurrentPlayer.Played)
            &&  instance.CurrentPlayer.Power == other.CurrentPlayer.Power
            &&  instance.CurrentPlayer.Prestige == other.CurrentPlayer.Prestige
                // Enemy player
            &&  instance.EnemyPlayer.Agents.IsIdentical(other.EnemyPlayer.Agents)
            &&  instance.EnemyPlayer.Coins == other.EnemyPlayer.Coins
            &&  instance.EnemyPlayer.CooldownPile.IsIdentical(other.EnemyPlayer.CooldownPile)
            &&  instance.EnemyPlayer.DrawPile.IsIdentical(other.EnemyPlayer.DrawPile)
            &&  instance.EnemyPlayer.Hand.IsIdentical(other.EnemyPlayer.Hand)
            &&  instance.EnemyPlayer.KnownUpcomingDraws.IsIdentical(other.EnemyPlayer.KnownUpcomingDraws)
            &&  instance.EnemyPlayer.PatronCalls == instance.EnemyPlayer.PatronCalls
            &&  instance.EnemyPlayer.Played.IsIdentical(other.EnemyPlayer.Played)
            &&  instance.EnemyPlayer.Power == instance.EnemyPlayer.Power
            &&  instance.EnemyPlayer.Prestige == instance.EnemyPlayer.Prestige

            &&  instance.Patrons.IsIdentical(other.Patrons)
            &&  instance.PatronStates.IsIdentical(other.PatronStates)
            &&  instance.PendingChoice.IsIdentical(other.PendingChoice)
            &&  instance.StartOfNextTurnEffects.IsIdentical(other.StartOfNextTurnEffects)
            &&  instance.TavernAvailableCards.IsIdentical(other.TavernAvailableCards)
            &&  instance.TavernCards.IsIdentical(other.TavernCards)
            &&  instance.UpcomingEffects.IsIdentical(other.UpcomingEffects)
            );
    }
    public static bool IsIdentical(this SerializedChoice? instance, SerializedChoice? other) {
        if (instance == null) {
            if (other == null) {
                return true;
            }
            else {
                return false;
            }
        }
        else if (other == null) {
            return false;
        }
        else{
            return (
                    instance.ChoiceFollowUp == other.ChoiceFollowUp
                // &&  instance.Context == other.Context // content should not matter if you have the same options
                &&  instance.MaxChoices == other.MaxChoices
                &&  instance.MinChoices == other.MinChoices
                &&  instance.Type == other.Type
                &&  ((instance.Type == Choice.DataType.CARD && instance.PossibleCards.IsIdentical(other.PossibleCards)) || (instance.Type == Choice.DataType.EFFECT && instance.PossibleEffects.IsIdentical(other.PossibleEffects)))
            );
        }
    }

    public static bool IsIdentical(this List<UniqueBaseEffect> instance, List<UniqueBaseEffect> other) {
        List<UniqueEffect> instanceUniqueEffects = new List<UniqueEffect>();
        List<UniqueEffectOr> instanceUniqueEffectOrs = new List<UniqueEffectOr>();
        List<UniqueEffectComposite> instanceUniqueEffectComposites = new List<UniqueEffectComposite>();

        List<UniqueEffect> otherUniqueEffects = new List<UniqueEffect>();
        List<UniqueEffectOr> otherUniqueEffectOrs = new List<UniqueEffectOr>();
        List<UniqueEffectComposite> otherUniqueEffectComposites = new List<UniqueEffectComposite>();

        foreach(var currEffect in instance){
            var uniqueEffect = currEffect as UniqueEffect;
            var uniqueEffectOr = currEffect as UniqueEffectOr;
            var uniqueEffectComposite = currEffect as UniqueEffectComposite;

            if (uniqueEffect != null) {
                instanceUniqueEffects.Add(uniqueEffect);
            }
            else if (uniqueEffectOr != null) {
                instanceUniqueEffectOrs.Add(uniqueEffectOr);
            }
            else if (uniqueEffectComposite != null) {
                instanceUniqueEffectComposites.Add(uniqueEffectComposite);
            }
            else {
                throw new Exception("TROUBLE HANDLING EFFECT COMPARISON");
            }
        }

        foreach(var currEffect in other){
            var uniqueEffect = currEffect as UniqueEffect;
            var uniqueEffectOr = currEffect as UniqueEffectOr;
            var uniqueEffectComposite = currEffect as UniqueEffectComposite;

            if (uniqueEffect != null) {
                otherUniqueEffects.Add(uniqueEffect);
            }
            else if (uniqueEffectOr != null) {
                otherUniqueEffectOrs.Add(uniqueEffectOr);
            }
            else if (uniqueEffectComposite != null) {
                otherUniqueEffectComposites.Add(uniqueEffectComposite);
            }
            else {
                throw new Exception("TROUBLE HANDLING EFFECT COMPARISON");
            }
        }

        return (
                instanceUniqueEffects.IsIdentical(otherUniqueEffects)
            &&  instanceUniqueEffectOrs.IsIdentical(otherUniqueEffectOrs)
            &&  instanceUniqueEffectComposites.IsIdentical(otherUniqueEffectComposites)
        );
    }

    public static bool IsIdentical(this List<UniqueEffect> instance, List<UniqueEffect> other) {
        instance.OrderBy(effect => effect.Amount).ThenBy(effect => effect.Combo).ThenBy(effect => effect.ParentCard.CommonId).ThenBy(effect => effect.Type);
        other.OrderBy(effect => effect.Amount).ThenBy(effect => effect.Combo).ThenBy(effect => effect.ParentCard.CommonId).ThenBy(effect => effect.Type);

        if (instance.Count != other.Count) {
            return false;
        }

        for(int i = 0; i < instance.Count; i++) {
            if (
                    instance[i].Amount != other[i].Amount
                ||  instance[i].Combo != other[i].Combo
                ||  instance[i].ParentCard.CommonId != other[i].ParentCard.CommonId
                ||  instance[i].Type != other[i].Type
            ) {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdentical(this List<UniqueEffectOr> instance, List<UniqueEffectOr> other) {
        instance.OrderBy(effect => effect.Combo).ThenBy(effect => effect.ParentCard.CommonId);
        other.OrderBy(effect => effect.Combo).ThenBy(effect => effect.ParentCard.CommonId);

        if (instance.Count != other.Count) {
            return false;
        }

        for(int i = 0; i < instance.Count; i++) {
            if (
                    instance[i].Combo != other[i].Combo
                ||  instance[i].ParentCard.CommonId != other[i].ParentCard.CommonId
            ) {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdentical(this List<UniqueEffectComposite> instance, List<UniqueEffectComposite> other) {
        instance.OrderBy(effect => effect.ParentCard.CommonId);
        other.OrderBy(effect => effect.ParentCard.CommonId);

        if (instance.Count != other.Count) {
            return false;
        }

        for(int i = 0; i < instance.Count; i++) {
            if (instance[i].ParentCard.CommonId != other[i].ParentCard.CommonId) {
                return false;
            }
        }

        return true;
    }
    public static bool IsIdentical(this List<SerializedAgent> instance, List<SerializedAgent> other) {

        if(instance.Count != other.Count) {
            return false;
        }

        var orderedInstanceAgents = instance.OrderBy(agent => agent.RepresentingCard.CommonId).ThenBy(agent => agent.Activated).ThenBy(agent => agent.CurrentHp).ToList();
        var orderedOtherAgents = other.OrderBy(agent => agent.RepresentingCard.CommonId).ThenBy(agent => agent.Activated).ThenBy(agent => agent.CurrentHp).ToList();

        for(int i = 0; i < instance.Count; i++) {
            if (
                    orderedInstanceAgents[i].Activated != orderedOtherAgents[i].Activated
                ||  orderedInstanceAgents[i].CurrentHp != orderedOtherAgents[i].CurrentHp
                ||  orderedInstanceAgents[i].RepresentingCard.CommonId != orderedOtherAgents[i].RepresentingCard.CommonId
            ) {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdentical(this ComboStates instance, ComboStates other) {
        if (instance.All.Count != other.All.Count) {
            return false;
        } 
        var instanceComboKeys = instance.All.OrderBy(state => state.Key).Select(state => state.Key);
        var otherComboKeys = other.All.OrderBy(state => state.Key).Select(state => state.Key);
        if (!instanceComboKeys.SequenceEqual(otherComboKeys)) {
            return false;
        }
        
        foreach(var currKey in instanceComboKeys) {
            // Since we also check which cards a played (which we need cause cards in played can be destroyed), we do not need to check which combo effects are ready for each patron
            // Since these will be equal if the cards played are equal
            if (instance.All[currKey].CurrentCombo != other.All[currKey].CurrentCombo) {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdentical(this List<UniqueCard> instance, List<UniqueCard> other) {
        // Unfortunately this is imprecise on knockout effect, since the SoT framework uses UniqueCard instead of agent in the choices, so i cant check CurrentHP, meaning that destroying an agent of type A
        // with full HP will be considered equal to destroying an agent of the same type with lower health
        List<CardId> orderedInstanceCardIds = instance.OrderBy(card => card.CommonId).Select(card => card.CommonId).ToList();
        List<CardId> orderedOtherCardIds = other.OrderBy(card => card.CommonId).Select(card => card.CommonId).ToList();
        return orderedInstanceCardIds.SequenceEqual(orderedOtherCardIds);
    }

    public static bool IsIdentical(this List<PatronId> instance, List<PatronId> other) {
        var orderedInstancePatronIds = instance.OrderBy(id => id);
        var orderedOtherPatronIds = other.OrderBy(id => id);

        return orderedInstancePatronIds.SequenceEqual(orderedOtherPatronIds);
    }

    public static bool IsIdentical(this PatronStates instance, PatronStates other) {

        if (instance.All.Keys.Count != other.All.Keys.Count) {
            return false;
        }

        var instanceKeysOrdered = instance.All.Keys.OrderBy(patronId => patronId).ToList();
        var otherKeysOrdered = instance.All.Keys.OrderBy(patronId => patronId).ToList();

        // This loop replaces list.sequenceEquals, cause for some reason that method wrongly returns false here
        for(int i = 0; i < instanceKeysOrdered.Count; i++) {
            if (instanceKeysOrdered[i] != otherKeysOrdered[i]){
                Console.WriteLine("we hit this cause: " + instanceKeysOrdered[i] + " != " + otherKeysOrdered[i]);
                return false;
            }
        }

        foreach(var patron in other.All.Keys) {
            if (instance.All[patron] != other.All[patron]) {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdentical(this Move instance, Move other) {
        if(instance.GetType() != other.GetType()) {
            return false;
        }

        switch(instance){
            case SimpleCardMove:
                return (
                        (instance as SimpleCardMove).Card.CommonId == (other as SimpleCardMove).Card.CommonId
                    &&  instance.Command == other.Command);
            case SimplePatronMove:
                return (
                    (instance as SimplePatronMove).PatronId == (other as SimplePatronMove).PatronId
                    && (instance as SimplePatronMove).Command == (other as SimplePatronMove).Command
                );
            case MakeChoiceMoveUniqueCard:
                return (
                    (instance as MakeChoiceMoveUniqueCard).Command == (other as MakeChoiceMoveUniqueCard).Command
                    &&  (instance as MakeChoiceMoveUniqueCard).Choices.IsIdentical((other as MakeChoiceMoveUniqueCard).Choices)
                );
            case MakeChoiceMoveUniqueEffect:
                return (
                    (instance as MakeChoiceMoveUniqueEffect).Command == (other as MakeChoiceMoveUniqueEffect).Command
                    && (instance as MakeChoiceMoveUniqueEffect).Choices.IsIdentical((other as MakeChoiceMoveUniqueEffect).Choices)
                );
            default:
                return instance.Command == other.Command;
        }
    }
}