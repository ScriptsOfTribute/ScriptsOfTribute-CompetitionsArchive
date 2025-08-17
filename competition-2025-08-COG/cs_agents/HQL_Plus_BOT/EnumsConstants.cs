using ScriptsOfTribute;

namespace Bots;


public class HQL_Plus_FilePaths
{
    private static readonly string BaseDir = AppContext.BaseDirectory;
    private static readonly string BotsDir = Path.Combine(BaseDir, "Bots");

    public static readonly string qTable1Path = Path.Combine(BotsDir, "QTABLE1.txt");
    public static readonly string qTable2Path = Path.Combine(BotsDir, "QTABLE2.txt");
    public static readonly string tmpFile     = Path.Combine(BotsDir, "tmp_file.txt");
    public static readonly string errorFile   = Path.Combine(BotsDir, "error_file.txt");
}

public static class HQL_Plus_Consts
{
    public static int number_of_all_cards = 125;
    public static int max_stage = 3;
    public static int max_combo = 3;
    public static int max_deck_cards = 19;

    public static int[] prestige_weight = new int[] { 15, 16, 20, 22, 100 };
    public static int[] power_weight = new int[] { 14, 15, 19, 19, 20 };
    public static int[] coins_weight = new int[] { 2, 1, 1, 1, 0 };
    public static int[] patron_weight = new int[] { 18, 18, 18, 18, 14 };
    public static int[] neutral_patron_weight = new int[] { 2, 2, 2, 2, 1 };
    public static int[] card_weight = new int[] { 5, 5, 4, 4, 2 };
    public static int[] combo_weight = new int[] { 2, 2, 2, 2, 1 };
    public static int[] active_agent_weight = new int[] { 12, 12, 12, 12, 12 };
    public static int[] hp_weight = new int[] { 4, 4, 4, 4, 3 };
    public static int[] ansei_weight = new int[] { 3, 2, 1, 1, 1 };
    public static int[] crow_weight = new int[] { -1000, -1000, 0, 20, 20 };
    public static int[] orgnum_weight = new int[] { 1, 1, 3, 5, 5 };
    public static int[] gold_card_weight = new int[] { 0, -2, -4, -6, -8 };

    public static double double_q_learning_alpha = 0.5; //probability of updating table 1 vs table 2
}

public enum HQL_Plus_Stage
{
    Start = 0,
    Early = 1,
    Middle = 2,
    Late = 3,
    End = 4,
}
