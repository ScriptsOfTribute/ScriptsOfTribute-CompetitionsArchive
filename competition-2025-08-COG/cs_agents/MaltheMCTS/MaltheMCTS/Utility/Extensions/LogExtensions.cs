using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;
using System.Text;

namespace MaltheMCTS;

public static class LogExtensions{

    public static string GetLog(this SeededGameState gameState)
    {
        var sb = new StringBuilder();
        using var writer = new StringWriter(sb);
        var originalOut = Console.Out;

        try
        {
            Console.SetOut(writer);
            gameState.Log();
        }
        finally
        {
            Console.SetOut(originalOut); // always resets out to be the console
        }

        return sb.ToString();
    }
    public static void Log(this SeededGameState gameState)
    {
        Console.WriteLine("GameState " + "(" + gameState.GenerateHash() + "):-----------------------------");
        Console.WriteLine("Combo states:");
        gameState.ComboStates.Log();
        Console.WriteLine("Current Player:");
        gameState.CurrentPlayer.Log();
        Console.WriteLine("---");
        Console.WriteLine("Enemy Player:");
        gameState.EnemyPlayer.Log();
        Console.WriteLine("---");
        Console.WriteLine("Patron states:");
        gameState.PatronStates.Log();
        Console.WriteLine("Pending choice:");
        gameState.PendingChoice?.Log();
        Console.WriteLine("Start of next turn effects:");
        gameState.StartOfNextTurnEffects.Log();
        Console.WriteLine("Tavern available cards:");
        gameState.TavernAvailableCards.Log();
        Console.WriteLine("Tavern cards size: " + gameState.TavernCards.Count);
        Console.WriteLine("Upcoming effects:");
        gameState.UpcomingEffects.Log();
        Console.WriteLine("Last completed action:");
        gameState.CompletedActions.Last().Log();
        Console.WriteLine("-----------------------------------------------------");
    }

    public static void Log(this CompletedAction completedAction)
    {
        Console.WriteLine("Type: " + completedAction.Type);
        Console.WriteLine("Player: " + completedAction.Player);
        Console.WriteLine("Type: " + completedAction.Type);
        Console.WriteLine("Source Card: " + (completedAction.SourceCard?.CommonId.ToString() ?? "null"));
        Console.WriteLine("Target Card: " + (completedAction.TargetCard?.CommonId.ToString() ?? "null"));
        Console.WriteLine("Source Patron: " + (completedAction.SourcePatron?.ToString() ?? "null"));
        Console.WriteLine("Combo: " + completedAction.Combo);
        Console.WriteLine("Amount: " + completedAction.Amount);
    }

    public static void Log(this List<UniqueBaseEffect> list){
        // Simplified for now. Expand if neccessary
        Console.WriteLine("Amount: " + list.Count);
    }

    public static void Log (this List<UniqueCard> list) {
        list.ForEach(card => Console.WriteLine(card.CommonId));
    }

    public static void Log(this SerializedChoice choice) {
        if (choice == null) {
            Console.WriteLine("N/A");
            return;
        }

        Console.WriteLine("Choice follow up: " + choice.ChoiceFollowUp);
        Console.WriteLine("Max choices: " + choice.MaxChoices);
        Console.WriteLine("Min choices: " + choice.MinChoices);
        Console.WriteLine("Type: " + choice.Type);

        switch(choice.Type) {
            case Choice.DataType.EFFECT:
                Console.WriteLine("Possible effects:");
                choice.PossibleEffects.ForEach(effect => {
                    Console.WriteLine("Parent card: " + effect.ParentCard.CommonId);
                    Console.WriteLine("Amount: " + effect.Amount);
                    Console.WriteLine("Combo: " + effect.Combo);
                });
                break;
            case Choice.DataType.CARD:
                Console.WriteLine("Possible cards:");
                choice.PossibleCards.ForEach(card => {
                    Console.WriteLine(card.CommonId);
                });
                break;
        }
    }

    public static void Log(this PatronStates patronStates) {
        foreach(var currKey in patronStates.All.Keys) {
            Console.WriteLine(currKey + ":");
            Console.WriteLine(patronStates.All[currKey]);
        }
    }

    public static void Log(this ComboStates comboStates) {
        foreach(var currKey in comboStates.All.Keys) {
            Console.WriteLine(currKey + ":");
            Console.WriteLine("Current combo: " + comboStates.All[currKey].CurrentCombo);
        }
    }

    public static void Log(this SerializedPlayer player)
    {
        Console.WriteLine("Coins: " + player.Coins);
        Console.WriteLine("Power: " + player.Power);
        Console.WriteLine("Prestige: " + player.Prestige);
        Console.WriteLine("Patron calls: " + player.PatronCalls);
        Console.WriteLine("Cards in hand:");
        player.Hand.ForEach(h => Console.WriteLine(h.CommonId));
        Console.WriteLine("Played cards:");
        player.Played.ForEach(h => Console.WriteLine(h.CommonId));
        Console.WriteLine("Known upcoming draws:");
        player.KnownUpcomingDraws.ForEach(h => Console.WriteLine(h.CommonId));
        Console.WriteLine("Cooldown size: " + player.CooldownPile.Count);
        Console.WriteLine("Drawpile size: " + player.DrawPile.Count);
        Console.WriteLine("Agents:");
        player.Agents.Log();
    }

    public static void Log(this List<SerializedAgent> list){
        list.ForEach(agent => {
            Console.WriteLine(agent.RepresentingCard.CommonId + ":");
            Console.WriteLine("Activated: " + agent.Activated);
            Console.WriteLine("HP: " + agent.CurrentHp);
        });
    }

    public static void Log(this Move move)
    {

        Console.WriteLine("Command:" + move.Command);
        Console.WriteLine("Type: " + move.GetType().Name);
        switch (move)
        {
            case SimpleCardMove simpleCardMove:
                Console.WriteLine("Card: " + simpleCardMove.Card.CommonId);
                break;
            case SimplePatronMove simplePatronMove:
                Console.WriteLine(simplePatronMove.PatronId);
                break;
            case MakeChoiceMoveUniqueCard uniqueCardMove:
                Console.WriteLine("Choices:");
                uniqueCardMove.Choices.ForEach(c => Console.Write(" " + c.CommonId + ","));
                break;
            case MakeChoiceMoveUniqueEffect uniqueEffectMove:
                Console.WriteLine("Choices:");
                uniqueEffectMove.Choices.ForEach(e => e.Log());
                break;
            default:
                Console.WriteLine("Unknown subtype");
                break;
        }
    }

    public static string GetLog(this Move move)
    {

        var log = "Command:" + move.Command;
        log += "Type: " + move.GetType().Name;
        switch (move)
        {
            case SimpleCardMove simpleCardMove:
                log += "Card: " + simpleCardMove.Card.CommonId;
                break;
            case SimplePatronMove simplePatronMove:
                log += simplePatronMove.PatronId;
                break;
            case MakeChoiceMoveUniqueCard uniqueCardMove:
                log += "Choices:";
                uniqueCardMove.Choices.ForEach(c => log += (" " + c.CommonId + ","));
                break;
            case MakeChoiceMoveUniqueEffect uniqueEffectMove:
                log += "Choices:";
                uniqueEffectMove.Choices.ForEach(e => e.GetLog());
                break;
            default:
                log += "Unknown subtype";
                break;
        }

        return log;
    }


    public static void Log(this UniqueEffect effect) {
        Console.WriteLine("Type: " + effect.Type);
        Console.WriteLine("Amount: " + effect.Amount);
        Console.WriteLine("Combo: " + effect.Combo);
        Console.WriteLine("Parent card: " + effect.ParentCard.CommonId);
    }

    public static string GetLog(this UniqueEffect effect) {
        var log = "Type: " + effect.Type;
        log += "Amount: " + effect.Amount;
        log += "Combo: " + effect.Combo;
        log += "Parent card: " + effect.ParentCard.CommonId;

        return log;
    }
}