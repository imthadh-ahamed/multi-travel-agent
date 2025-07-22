"""
Phase 1: Foundation - Rule-Based Travel Advisor

This module implements a simple rule-based travel advisor that recommends
destinations based on user preferences like budget, weather, and region.

Learning Objectives:
- Understand basic agent concepts
- Implement rule-based decision making
- Create a simple CLI interface
- Handle user input and preferences
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Budget(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Climate(Enum):
    TROPICAL = "tropical"
    TEMPERATE = "temperate"
    COLD = "cold"
    DESERT = "desert"


class Region(Enum):
    EUROPE = "europe"
    ASIA = "asia"
    AMERICA = "america"
    AFRICA = "africa"
    OCEANIA = "oceania"


@dataclass
class Destination:
    name: str
    country: str
    region: Region
    climate: Climate
    budget_level: Budget
    attractions: List[str]
    best_months: List[str]
    description: str


@dataclass
class UserPreferences:
    budget: Budget
    preferred_climate: Climate
    preferred_region: Optional[Region] = None
    travel_months: Optional[List[str]] = None
    interests: Optional[List[str]] = None


class RuleBasedTravelAdvisor:
    """
    A rule-based agent that recommends travel destinations based on user preferences.
    Uses simple conditional logic to match destinations with user requirements.
    """
    
    def __init__(self):
        self.destinations = self._load_destinations()
        
    def _load_destinations(self) -> List[Destination]:
        """Load sample destination data"""
        return [
            Destination(
                name="Bali, Indonesia",
                country="Indonesia",
                region=Region.ASIA,
                climate=Climate.TROPICAL,
                budget_level=Budget.MEDIUM,
                attractions=["Beaches", "Temples", "Rice Terraces", "Volcanoes"],
                best_months=["Apr", "May", "Jun", "Jul", "Aug", "Sep"],
                description="Tropical paradise with rich culture and stunning landscapes"
            ),
            Destination(
                name="Prague, Czech Republic",
                country="Czech Republic",
                region=Region.EUROPE,
                climate=Climate.TEMPERATE,
                budget_level=Budget.LOW,
                attractions=["Historic Architecture", "Museums", "Beer Culture", "Castles"],
                best_months=["May", "Jun", "Jul", "Aug", "Sep"],
                description="Beautiful historic city with affordable prices"
            ),
            Destination(
                name="Tokyo, Japan",
                country="Japan",
                region=Region.ASIA,
                climate=Climate.TEMPERATE,
                budget_level=Budget.HIGH,
                attractions=["Technology", "Culture", "Food", "Shopping"],
                best_months=["Mar", "Apr", "May", "Oct", "Nov"],
                description="Modern metropolis blending tradition and innovation"
            ),
            Destination(
                name="Reykjavik, Iceland",
                country="Iceland",
                region=Region.EUROPE,
                climate=Climate.COLD,
                budget_level=Budget.HIGH,
                attractions=["Northern Lights", "Geysers", "Waterfalls", "Blue Lagoon"],
                best_months=["Jun", "Jul", "Aug"],
                description="Land of fire and ice with unique natural phenomena"
            ),
            Destination(
                name="Marrakech, Morocco",
                country="Morocco",
                region=Region.AFRICA,
                climate=Climate.DESERT,
                budget_level=Budget.LOW,
                attractions=["Medina", "Souks", "Architecture", "Desert Tours"],
                best_months=["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"],
                description="Exotic destination with rich history and vibrant culture"
            ),
            Destination(
                name="Sydney, Australia",
                country="Australia",
                region=Region.OCEANIA,
                climate=Climate.TEMPERATE,
                budget_level=Budget.HIGH,
                attractions=["Opera House", "Beaches", "Harbour Bridge", "Wildlife"],
                best_months=["Dec", "Jan", "Feb", "Mar"],
                description="Iconic city with beautiful harbors and laid-back culture"
            )
        ]
    
    def recommend_destinations(self, preferences: UserPreferences) -> List[Destination]:
        """
        Apply rules to recommend destinations based on user preferences
        """
        recommendations = []
        
        for destination in self.destinations:
            score = self._calculate_match_score(destination, preferences)
            if score > 0.5:  # Threshold for recommendation
                recommendations.append((destination, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [dest for dest, score in recommendations[:5]]  # Top 5 recommendations
    
    def _calculate_match_score(self, destination: Destination, preferences: UserPreferences) -> float:
        """
        Calculate how well a destination matches user preferences
        """
        score = 0.0
        total_weight = 0.0
        
        # Budget matching (weight: 0.3)
        if destination.budget_level == preferences.budget:
            score += 0.3
        total_weight += 0.3
        
        # Climate matching (weight: 0.4)
        if destination.climate == preferences.preferred_climate:
            score += 0.4
        total_weight += 0.4
        
        # Region matching (weight: 0.2)
        if preferences.preferred_region and destination.region == preferences.preferred_region:
            score += 0.2
        total_weight += 0.2
        
        # Travel month matching (weight: 0.1)
        if preferences.travel_months:
            month_match = any(month in destination.best_months for month in preferences.travel_months)
            if month_match:
                score += 0.1
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_detailed_recommendation(self, destination: Destination, preferences: UserPreferences) -> str:
        """
        Generate a detailed recommendation explanation
        """
        reasons = []
        
        if destination.budget_level == preferences.budget:
            reasons.append(f"Matches your {preferences.budget.value} budget")
        
        if destination.climate == preferences.preferred_climate:
            reasons.append(f"Perfect {preferences.preferred_climate.value} climate")
        
        if preferences.preferred_region and destination.region == preferences.preferred_region:
            reasons.append(f"Located in your preferred region: {preferences.preferred_region.value}")
        
        if preferences.travel_months:
            matching_months = [m for m in preferences.travel_months if m in destination.best_months]
            if matching_months:
                reasons.append(f"Great time to visit in: {', '.join(matching_months)}")
        
        recommendation = f"""
🌍 {destination.name}
📍 {destination.country} ({destination.region.value.title()})
💰 Budget Level: {destination.budget_level.value.title()}
🌡️ Climate: {destination.climate.value.title()}

📝 Description: {destination.description}

🎯 Top Attractions:
{chr(10).join(f"  • {attraction}" for attraction in destination.attractions)}

📅 Best Months: {', '.join(destination.best_months)}

✅ Why This Matches You:
{chr(10).join(f"  • {reason}" for reason in reasons)}
        """
        
        return recommendation.strip()


def get_user_preferences() -> UserPreferences:
    """
    Interactive CLI to gather user preferences
    """
    print("🌍 Welcome to the Rule-Based Travel Advisor! 🌍")
    print("Let's find your perfect destination based on your preferences.\n")
    
    # Budget
    print("💰 What's your budget level?")
    print("1. Low (Budget-friendly)")
    print("2. Medium (Moderate spending)")
    print("3. High (Luxury travel)")
    
    budget_choice = input("Enter choice (1-3): ").strip()
    budget_map = {"1": Budget.LOW, "2": Budget.MEDIUM, "3": Budget.HIGH}
    budget = budget_map.get(budget_choice, Budget.MEDIUM)
    
    # Climate
    print("\n🌡️ What climate do you prefer?")
    print("1. Tropical (Warm and humid)")
    print("2. Temperate (Mild seasons)")
    print("3. Cold (Cool/cold weather)")
    print("4. Desert (Dry and warm)")
    
    climate_choice = input("Enter choice (1-4): ").strip()
    climate_map = {
        "1": Climate.TROPICAL,
        "2": Climate.TEMPERATE,
        "3": Climate.COLD,
        "4": Climate.DESERT
    }
    climate = climate_map.get(climate_choice, Climate.TEMPERATE)
    
    # Region (optional)
    print("\n🗺️ Any preferred region? (optional)")
    print("1. Europe")
    print("2. Asia")
    print("3. America")
    print("4. Africa")
    print("5. Oceania")
    print("6. No preference")
    
    region_choice = input("Enter choice (1-6): ").strip()
    region_map = {
        "1": Region.EUROPE,
        "2": Region.ASIA,
        "3": Region.AMERICA,
        "4": Region.AFRICA,
        "5": Region.OCEANIA,
        "6": None
    }
    region = region_map.get(region_choice, None)
    
    # Travel months (optional)
    print("\n📅 When are you planning to travel? (optional)")
    print("Enter months (e.g., 'Jan, Feb, Mar') or press Enter to skip:")
    months_input = input().strip()
    travel_months = None
    if months_input:
        travel_months = [month.strip() for month in months_input.split(",")]
    
    return UserPreferences(
        budget=budget,
        preferred_climate=climate,
        preferred_region=region,
        travel_months=travel_months
    )


def main():
    """
    Main CLI application
    """
    try:
        # Initialize the agent
        advisor = RuleBasedTravelAdvisor()
        
        # Get user preferences
        preferences = get_user_preferences()
        
        # Get recommendations
        print("\n🔍 Analyzing your preferences and finding perfect destinations...\n")
        recommendations = advisor.recommend_destinations(preferences)
        
        if not recommendations:
            print("❌ Sorry, no destinations match your exact preferences.")
            print("Try adjusting your criteria for more options.")
            return
        
        print(f"✅ Found {len(recommendations)} perfect destinations for you!\n")
        print("=" * 60)
        
        # Display recommendations
        for i, destination in enumerate(recommendations, 1):
            print(f"\n🏆 RECOMMENDATION #{i}")
            print("=" * 60)
            detailed_rec = advisor.get_detailed_recommendation(destination, preferences)
            print(detailed_rec)
            
            if i < len(recommendations):
                print("\n" + "-" * 60)
        
        print(f"\n🎉 Happy travels! These {len(recommendations)} destinations are perfect for you!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Travel Advisor! Safe travels!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again.")


if __name__ == "__main__":
    main()
