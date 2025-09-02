using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

namespace MaltheMCTS;

public static class HashExtensions
{
    public static int GenerateHash(this SeededGameState state)
    {
        var hashCode = new HashCode();

        int comboHash = 1;

        foreach (var comboState in state.ComboStates.All)
        {
            comboHash *= ((int)comboState.Key + 1) * 10;
            comboHash *= (comboState.Value.CurrentCombo + 1) * 2;
        }

        hashCode.Add(comboHash);

        hashCode.Add(state.CurrentPlayer.GenerateHash());
        hashCode.Add(state.EnemyPlayer.GenerateHash());

        int patronHash = 1;

        foreach (var currPatron in state.Patrons)
        {
            patronHash *= (((int)currPatron + 1) * 10);
        }

        hashCode.Add(patronHash);

        int patronStateHash = 1;

        foreach (var patronState in state.PatronStates.All)
        {
            patronStateHash *= (((int)patronState.Key + 1) * 200);
            patronStateHash *= ((int)patronState.Value + 1);
        }

        hashCode.Add(patronStateHash);

        int PendingChoiceHash = 1;

        if (state.PendingChoice != null)
        {
            PendingChoiceHash *= ((int)state.PendingChoice.ChoiceFollowUp + 1) * 100;
            PendingChoiceHash *= state.PendingChoice.MaxChoices + 1;
            PendingChoiceHash *= state.PendingChoice.MinChoices + 1;
            PendingChoiceHash *= ((int)state.PendingChoice.Type + 1);

            switch (state.PendingChoice.Type)
            {
                case Choice.DataType.CARD:
                    foreach (var currCard in state.PendingChoice.PossibleCards)
                    {
                        PendingChoiceHash *= ((int)currCard.UniqueId + 1) * 100;
                    }
                    break;
                case Choice.DataType.EFFECT:
                    foreach (var currEffect in state.PendingChoice.PossibleEffects)
                    {
                        PendingChoiceHash *= (currEffect.Amount + 1) * 100;
                        PendingChoiceHash *= (currEffect.Combo + 1) * 1_000;
                        PendingChoiceHash *= ((int)currEffect.Type + 1);
                        PendingChoiceHash *= ((int)currEffect.ParentCard.CommonId + 1);
                    }
                    break;
            }
        }

        hashCode.Add(PendingChoiceHash);

        int StartOfNextTurnEffectsHash = 1;

        foreach (var currEffect in state.StartOfNextTurnEffects)
        {
            StartOfNextTurnEffectsHash *= currEffect.GenerateHash();
        }

        hashCode.Add(StartOfNextTurnEffectsHash);

        int tavernAvailAbleCardsHash = 1;

        foreach (var currCard in state.TavernAvailableCards)
        {
            tavernAvailAbleCardsHash *= (int)currCard.CommonId + 1;
        }

        hashCode.Add(tavernAvailAbleCardsHash);

        int tavernCardsHash = 1;

        foreach (var currCard in state.TavernCards)
        {
            tavernCardsHash *= ((int)currCard.CommonId + 1) * 100;
        }

        hashCode.Add(tavernCardsHash);

        int upcomingEffectsHash = 1;

        foreach (var currEffect in state.UpcomingEffects)
        {
            upcomingEffectsHash *= currEffect.GenerateHash();
        }

        hashCode.Add(upcomingEffectsHash);

        return hashCode.ToHashCode();
    }

    public static int GenerateHash(this UniqueBaseEffect effect)
    {

        var hashCode = new HashCode();

        var uniqueEffect = effect as UniqueEffect;
        var uniqueEffectOr = effect as UniqueEffectOr;
        var uniqueEffectComposite = effect as UniqueEffectComposite;

        if (uniqueEffect != null)
        {
            hashCode.Add(uniqueEffect.Amount);
            hashCode.Add(uniqueEffect.Combo * 100);
            hashCode.Add(((int)uniqueEffect.ParentCard.UniqueId) * 1_000);
        }
        else if (uniqueEffectOr != null)
        {
            hashCode.Add(uniqueEffectOr.Combo * 10_000);
            hashCode.Add(((int)uniqueEffectOr.ParentCard.CommonId) * 100_000);
        }
        else if (uniqueEffectComposite != null)
        {
            hashCode.Add(((int)uniqueEffectComposite.ParentCard.CommonId) * 1_000_000);
        }

        return hashCode.ToHashCode();
    }

    public static int GenerateHash(this SerializedPlayer player)
    {

        var hashCode = new HashCode();

        int agentsHash = 1;

        foreach (var currAgent in player.Agents)
        {
            agentsHash *= currAgent.Activated ? 1 : 100;
            agentsHash *= currAgent.CurrentHp + 1_000;
            agentsHash *= ((int)currAgent.RepresentingCard.CommonId + 1);
        }

        hashCode.Add(agentsHash);
        hashCode.Add(player.Coins);

        int cooldownHash = 1;

        foreach (var currCard in player.CooldownPile)
        {
            cooldownHash *= ((int)currCard.CommonId + 1) * 100;
        }

        hashCode.Add(cooldownHash);

        int drawPileHash = 1;

        foreach (var currCard in player.DrawPile)
        {
            drawPileHash *= ((int)currCard.CommonId+1) * 1000;
        }

        hashCode.Add(drawPileHash);

        int knownUpcomingHash = 1;

        foreach (var currCard in player.KnownUpcomingDraws)
        {
            knownUpcomingHash *= ((int)currCard.CommonId + 1) * 10_000;
        }

        hashCode.Add(knownUpcomingHash);
        hashCode.Add(player.PatronCalls);

        int playedHash = 1;

        foreach (var currCard in player.Played)
        {
            playedHash *= ((int)currCard.CommonId + 1);
        }

        hashCode.Add(playedHash);
        hashCode.Add(player.Power);
        hashCode.Add(player.Prestige);

        return hashCode.ToHashCode();
    }
}