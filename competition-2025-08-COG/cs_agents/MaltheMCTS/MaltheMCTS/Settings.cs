using System.Reflection;

namespace MaltheMCTS;

public class Settings
{
    public double ITERATION_COMPLETION_MILLISECONDS_BUFFER { get; set; }
    public double UCT_EXPLORATION_CONSTANT { get; set; }
    public bool FORCE_DELAY_TURN_END_IN_ROLLOUT { get; set; }
    public bool INCLUDE_PLAY_MOVE_CHANCE_NODES { get; set; }
    public bool INCLUDE_END_TURN_CHANCE_NODES { get; set; }
    public int? CHANCE_NODE_BRANCH_LIMIT { get; set; }
    public int ROLLOUT_TURNS_BEFORE_HEURISTIC { get; set; }
    public bool REUSE_TREE { get; set; }
    public bool UPDATED_TREE_REUSE { get; set; }
    public bool SIMULATE_MULTIPLE_TURNS { get; set; }
    public int? CHOICE_BRANCH_LIMIT { get; set; }
    public bool ADDITIONAL_MOVE_FILTERING { get; set; }

    public Settings()
    {
        ITERATION_COMPLETION_MILLISECONDS_BUFFER = 50;
        UCT_EXPLORATION_CONSTANT = 1.41421356237; // sqrt(2) generally used default value
        FORCE_DELAY_TURN_END_IN_ROLLOUT = true;
        INCLUDE_PLAY_MOVE_CHANCE_NODES = false;
        INCLUDE_END_TURN_CHANCE_NODES = false;
        CHANCE_NODE_BRANCH_LIMIT = 3;
            ROLLOUT_TURNS_BEFORE_HEURISTIC = 1;
        REUSE_TREE = true;
            UPDATED_TREE_REUSE = true;
        SIMULATE_MULTIPLE_TURNS = false;
        CHOICE_BRANCH_LIMIT = 5;
        ADDITIONAL_MOVE_FILTERING = true;
    }

    public override string ToString()
    {
        var props = GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public);
        return string.Join("\n", props.Select(p => $"{p.Name}={p.GetValue(this)}"));
    }

    public static Settings LoadFromFile(string filePath)
    {
        var result = new Settings();
        var lines = File.ReadAllLines(filePath);

        foreach (var line in lines)
        {
            var parts = line.Split('=');
            if (parts.Length == 2)
            {
                var key = parts[0].Trim();
                var value = parts[1].Trim();

                var property = typeof(Settings).GetProperty(key);
                if (property != null)
                {
                    if (value.Equals("null", StringComparison.OrdinalIgnoreCase))
                    {
                        property.SetValue(result, null);
                    }
                    else if (property.PropertyType == typeof(int) || property.PropertyType == typeof(int?))
                    {
                        property.SetValue(result, int.Parse(value));
                    }
                    else if (property.PropertyType == typeof(double) || property.PropertyType == typeof(double?))
                    {
                        property.SetValue(result, double.Parse(value));
                    }
                    else if (property.PropertyType == typeof(bool) || property.PropertyType == typeof(bool?))
                    {
                        property.SetValue(result, bool.Parse(value));
                    }
                    else if (property.PropertyType.IsEnum)
                    {
                        var enumValue = Enum.Parse(property.PropertyType, value);
                        property.SetValue(result, enumValue);
                    }
                    else if (Nullable.GetUnderlyingType(property.PropertyType)?.IsEnum == true)
                    {
                        var enumType = Nullable.GetUnderlyingType(property.PropertyType);
                        var enumValue = Enum.Parse(enumType, value);
                        property.SetValue(result, enumValue);
                    }
                }
            }
        }

        return result;
    }
}