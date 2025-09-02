using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring
{
    public struct CardStrengths
    {
        public double PrestigeStrength = 0;
        public double PowerStrength = 0;
        public double GoldStrength = 0;
        public double MiscellaneousStrength = 0;
        // Consider if i should consider other overall properties than these four
        public CardStrengths()
        {
        }

        public static CardStrengths operator +(CardStrengths a, CardStrengths b)
        {
            var prestigeStrength = a.PrestigeStrength + b.PrestigeStrength;
            var powerStrength = a.PowerStrength + b.PowerStrength;
            var goldStrength = a.GoldStrength + b.GoldStrength;
            var miscellaneousStrength = a.MiscellaneousStrength + b.MiscellaneousStrength;
            return new CardStrengths()
            {
                PrestigeStrength = prestigeStrength,
                PowerStrength = powerStrength,
                GoldStrength = goldStrength,
                MiscellaneousStrength = miscellaneousStrength
            };
        }

        public static CardStrengths operator *(CardStrengths a, double multiplier)
        {
            return new CardStrengths()
            {
                PrestigeStrength = a.PrestigeStrength * multiplier,
                PowerStrength = a.PowerStrength * multiplier,
                GoldStrength = a.GoldStrength * multiplier,
                MiscellaneousStrength = a.MiscellaneousStrength * multiplier
            };
        }

        public static CardStrengths operator /(CardStrengths a, int divisor)
        {
            return new CardStrengths()
            {
                PrestigeStrength = a.PrestigeStrength / divisor,
                PowerStrength = a.PowerStrength / divisor,
                GoldStrength = a.GoldStrength / divisor,
                MiscellaneousStrength = a.MiscellaneousStrength / divisor
            };
        }
    }
}
