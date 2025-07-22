"""
External API Tools for Travel Agents

This module provides comprehensive API integrations for travel-related services.
It includes real implementations and mock services for development and testing.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import openai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIResponse:
    """Standardized API response format"""
    success: bool
    data: Any
    error: Optional[str] = None
    status_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ExternalAPI(ABC):
    """Base class for external API integrations"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the API is accessible"""
        pass
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Optional[Dict] = None, 
                           data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request to the API"""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            # Add API key to headers or params
            headers = {"User-Agent": "MultiTravelAgent/1.0"}
            if not params:
                params = {}
            
            # Add API key (adapt based on API requirements)
            params["appid"] = self.api_key
            
            async with self.session.request(method, url, params=params, json=data, headers=headers) as response:
                response_data = await response.json()
                
                return APIResponse(
                    success=response.status == 200,
                    data=response_data if response.status == 200 else None,
                    error=f"HTTP {response.status}: {response_data.get('message', 'Unknown error')}" if response.status != 200 else None,
                    status_code=response.status,
                    metadata={"endpoint": endpoint, "method": method}
                )
                
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error=str(e),
                metadata={"endpoint": endpoint, "method": method}
            )


class WeatherAPIClient(ExternalAPI):
    """OpenWeatherMap API client"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "http://api.openweathermap.org/data/2.5")
    
    async def health_check(self) -> bool:
        """Check API health"""
        try:
            response = await self._make_request("GET", "/weather", {"q": "London", "units": "metric"})
            return response.success
        except:
            return False
    
    async def get_current_weather(self, city: str, country_code: str = "") -> APIResponse:
        """Get current weather for a city"""
        location = f"{city},{country_code}" if country_code else city
        return await self._make_request("GET", "/weather", {"q": location, "units": "metric"})
    
    async def get_forecast(self, city: str, days: int = 5, country_code: str = "") -> APIResponse:
        """Get weather forecast"""
        location = f"{city},{country_code}" if country_code else city
        return await self._make_request("GET", "/forecast", {
            "q": location,
            "units": "metric",
            "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
        })
    
    async def get_weather_by_coordinates(self, lat: float, lon: float) -> APIResponse:
        """Get weather by coordinates"""
        return await self._make_request("GET", "/weather", {
            "lat": lat,
            "lon": lon,
            "units": "metric"
        })


class CurrencyAPIClient:
    """Currency exchange rate API client (using free API)"""
    
    def __init__(self):
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
    
    async def get_exchange_rates(self, base_currency: str = "USD") -> APIResponse:
        """Get current exchange rates"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/{base_currency}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return APIResponse(success=True, data=data)
                    else:
                        return APIResponse(
                            success=False,
                            data=None,
                            error=f"HTTP {response.status}",
                            status_code=response.status
                        )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    async def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> APIResponse:
        """Convert currency"""
        try:
            rates_response = await self.get_exchange_rates(from_currency)
            if not rates_response.success:
                return rates_response
            
            rates = rates_response.data.get("rates", {})
            if to_currency not in rates:
                return APIResponse(
                    success=False,
                    data=None,
                    error=f"Currency {to_currency} not found"
                )
            
            converted_amount = amount * rates[to_currency]
            
            return APIResponse(
                success=True,
                data={
                    "original_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "converted_amount": round(converted_amount, 2),
                    "exchange_rate": rates[to_currency],
                    "date": rates_response.data.get("date")
                }
            )
            
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))


class NewsAPIClient:
    """News API client for travel news and updates"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    async def get_travel_news(self, country: str = "us", category: str = "general") -> APIResponse:
        """Get travel-related news"""
        try:
            params = {
                "apiKey": self.api_key,
                "country": country,
                "category": category,
                "q": "travel OR tourism OR vacation",
                "sortBy": "publishedAt",
                "pageSize": 10
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/top-headlines", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return APIResponse(success=True, data=data)
                    else:
                        return APIResponse(
                            success=False,
                            data=None,
                            error=f"HTTP {response.status}",
                            status_code=response.status
                        )
                        
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))


class MockFlightAPI:
    """Mock flight API for development and testing"""
    
    def __init__(self):
        self.airlines = [
            "SkyWings Airlines", "Global Airways", "Premium Air", "Budget Fly",
            "International Jets", "Comfort Airlines", "Swift Air", "Elite Wings"
        ]
        
        self.aircraft_types = [
            "Boeing 737", "Airbus A320", "Boeing 787", "Airbus A350",
            "Boeing 777", "Airbus A380", "Embraer E190", "Boeing 757"
        ]
    
    async def search_flights(self, origin: str, destination: str, 
                           departure_date: str, return_date: str = None,
                           passengers: int = 1, cabin_class: str = "economy") -> APIResponse:
        """Mock flight search"""
        try:
            await asyncio.sleep(0.5)  # Simulate API delay
            
            # Generate mock flights
            flights = []
            for i in range(3, 8):  # 3-7 flights
                # Base price calculation
                base_price = 200 + (i * 50) + (passengers * 100)
                if cabin_class == "business":
                    base_price *= 3
                elif cabin_class == "first":
                    base_price *= 5
                
                # Add some randomness
                import random
                price_variation = random.randint(-50, 100)
                final_price = max(base_price + price_variation, 100)
                
                flight = {
                    "airline": random.choice(self.airlines),
                    "flight_number": f"{random.choice(['SK', 'GA', 'PA', 'BF'])}{random.randint(100, 999)}",
                    "departure_time": f"{random.randint(6, 22):02d}:{random.choice(['00', '15', '30', '45'])}",
                    "arrival_time": f"{random.randint(8, 23):02d}:{random.choice(['00', '15', '30', '45'])}",
                    "duration": f"{random.randint(2, 15)}h {random.randint(0, 59)}m",
                    "price": final_price,
                    "currency": "USD",
                    "stops": random.choice([0, 0, 1, 2]),  # More direct flights
                    "aircraft": random.choice(self.aircraft_types),
                    "cabin_class": cabin_class,
                    "baggage_included": cabin_class != "economy" or random.choice([True, False]),
                    "refundable": cabin_class != "economy" or random.choice([True, False]),
                    "booking_class": random.choice(["Y", "B", "M", "H", "Q", "V", "W"])
                }
                
                flights.append(flight)
            
            # Sort by price
            flights.sort(key=lambda x: x["price"])
            
            result = {
                "search_params": {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "return_date": return_date,
                    "passengers": passengers,
                    "cabin_class": cabin_class
                },
                "flights": flights,
                "total_results": len(flights),
                "search_time": datetime.now().isoformat(),
                "currency": "USD"
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    async def get_flight_details(self, flight_number: str) -> APIResponse:
        """Get detailed flight information"""
        try:
            await asyncio.sleep(0.2)
            
            import random
            
            details = {
                "flight_number": flight_number,
                "airline": random.choice(self.airlines),
                "aircraft": random.choice(self.aircraft_types),
                "route": {
                    "origin": {"code": "JFK", "name": "John F. Kennedy International", "city": "New York"},
                    "destination": {"code": "LHR", "name": "London Heathrow", "city": "London"}
                },
                "schedule": {
                    "departure": f"{random.randint(6, 22):02d}:{random.choice(['00', '15', '30', '45'])}",
                    "arrival": f"{random.randint(8, 23):02d}:{random.choice(['00', '15', '30', '45'])}",
                    "duration": f"{random.randint(6, 12)}h {random.randint(0, 59)}m"
                },
                "aircraft_details": {
                    "model": random.choice(self.aircraft_types),
                    "total_seats": random.randint(150, 400),
                    "seat_configuration": random.choice(["3-3-3", "2-4-2", "3-4-3"]),
                    "wifi_available": random.choice([True, False]),
                    "entertainment": random.choice([True, False])
                },
                "baggage_policy": {
                    "carry_on": "1 piece, 8kg max",
                    "checked": "1 piece, 23kg max" if random.choice([True, False]) else "Not included",
                    "extra_baggage_fee": random.randint(50, 150)
                },
                "amenities": random.sample([
                    "In-flight entertainment", "WiFi", "Power outlets", "USB charging",
                    "Meals included", "Snacks", "Beverages", "Blanket and pillow"
                ], random.randint(3, 6))
            }
            
            return APIResponse(success=True, data=details)
            
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))


class MockHotelAPI:
    """Mock hotel API for development and testing"""
    
    def __init__(self):
        self.hotel_chains = [
            "Hilton", "Marriott", "Hyatt", "InterContinental", "Accor",
            "Radisson", "Best Western", "Holiday Inn", "Sheraton", "Westin"
        ]
        
        self.hotel_types = [
            "Luxury Hotel", "Business Hotel", "Boutique Hotel", "Resort",
            "Budget Hotel", "Extended Stay", "Apartment Hotel", "Spa Resort"
        ]
        
        self.amenities_pool = [
            "Free WiFi", "Swimming Pool", "Fitness Center", "Spa", "Restaurant",
            "Room Service", "Business Center", "Concierge", "Valet Parking",
            "Airport Shuttle", "Pet Friendly", "Laundry Service", "Conference Rooms",
            "Bar/Lounge", "24-hour Front Desk", "Air Conditioning", "Minibar",
            "Safe", "Balcony", "City View", "Ocean View", "Mountain View"
        ]
    
    async def search_hotels(self, city: str, check_in: str, check_out: str,
                           guests: int = 2, rooms: int = 1, 
                           max_price: float = None, min_rating: float = None) -> APIResponse:
        """Mock hotel search"""
        try:
            await asyncio.sleep(0.3)  # Simulate API delay
            
            import random
            
            # Generate mock hotels
            hotels = []
            for i in range(5, 12):  # 5-11 hotels
                # Base price calculation
                base_price = 80 + (i * 20) + (guests * 10)
                price_variation = random.randint(-30, 80)
                nightly_rate = max(base_price + price_variation, 50)
                
                # Calculate nights
                from datetime import datetime
                try:
                    checkin_date = datetime.strptime(check_in, "%Y-%m-%d")
                    checkout_date = datetime.strptime(check_out, "%Y-%m-%d")
                    nights = (checkout_date - checkin_date).days
                except:
                    nights = 3  # Default to 3 nights
                
                total_price = nightly_rate * nights * rooms
                
                # Apply max price filter
                if max_price and nightly_rate > max_price:
                    continue
                
                rating = round(random.uniform(3.0, 5.0), 1)
                
                # Apply min rating filter
                if min_rating and rating < min_rating:
                    continue
                
                hotel = {
                    "name": f"{random.choice(self.hotel_chains)} {random.choice(['Grand', 'Plaza', 'Suites', 'Resort', 'Inn', 'Hotel'])} {city}",
                    "type": random.choice(self.hotel_types),
                    "rating": rating,
                    "review_count": random.randint(100, 2000),
                    "price_per_night": nightly_rate,
                    "total_price": total_price,
                    "currency": "USD",
                    "location": {
                        "address": f"{random.randint(100, 999)} {random.choice(['Main', 'Central', 'Hotel', 'Tourism'])} Street",
                        "district": random.choice(['City Center', 'Downtown', 'Old Town', 'Business District', 'Tourist Area']),
                        "distance_to_center": f"{random.uniform(0.1, 5.0):.1f} km",
                        "distance_to_airport": f"{random.randint(5, 50)} km"
                    },
                    "amenities": random.sample(self.amenities_pool, random.randint(8, 15)),
                    "room_info": {
                        "room_type": random.choice(["Standard Room", "Deluxe Room", "Suite", "Executive Room"]),
                        "bed_type": random.choice(["Queen Bed", "King Bed", "Twin Beds", "Double Bed"]),
                        "room_size": f"{random.randint(20, 60)} sqm",
                        "max_occupancy": guests
                    },
                    "policies": {
                        "cancellation": random.choice([
                            "Free cancellation until 24h before arrival",
                            "Free cancellation until 48h before arrival",
                            "Non-refundable",
                            "Partially refundable"
                        ]),
                        "breakfast": random.choice(["Included", "Not included", "Optional ($15/day)"]),
                        "parking": random.choice(["Free", "Paid ($20/day)", "Valet ($35/day)", "Not available"]),
                        "pets": random.choice(["Allowed", "Not allowed", "Allowed with fee"])
                    },
                    "booking_info": {
                        "availability": "Available",
                        "instant_confirmation": random.choice([True, False]),
                        "pay_at_hotel": random.choice([True, False])
                    }
                }
                
                hotels.append(hotel)
            
            # Sort by rating then price
            hotels.sort(key=lambda x: (-x["rating"], x["price_per_night"]))
            
            result = {
                "search_params": {
                    "city": city,
                    "check_in": check_in,
                    "check_out": check_out,
                    "guests": guests,
                    "rooms": rooms,
                    "nights": nights
                },
                "hotels": hotels,
                "total_results": len(hotels),
                "search_time": datetime.now().isoformat(),
                "currency": "USD"
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    async def get_hotel_details(self, hotel_id: str) -> APIResponse:
        """Get detailed hotel information"""
        try:
            await asyncio.sleep(0.2)
            
            import random
            
            details = {
                "hotel_id": hotel_id,
                "name": f"{random.choice(self.hotel_chains)} Grand Plaza",
                "description": "Experience luxury and comfort at our premier hotel located in the heart of the city. Our elegant rooms and suites offer modern amenities and stunning views.",
                "category": random.choice(self.hotel_types),
                "star_rating": random.randint(3, 5),
                "guest_rating": round(random.uniform(7.5, 9.5), 1),
                "location": {
                    "latitude": random.uniform(-90, 90),
                    "longitude": random.uniform(-180, 180),
                    "address": "123 Main Street, City Center",
                    "postal_code": str(random.randint(10000, 99999)),
                    "phone": f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    "email": "info@hotel.com"
                },
                "facilities": {
                    "general": random.sample(self.amenities_pool, 10),
                    "room_features": random.sample([
                        "Air conditioning", "Minibar", "Safe", "Flat-screen TV",
                        "Coffee/tea maker", "Hairdryer", "Bathrobes", "Slippers",
                        "Work desk", "Seating area", "Balcony", "City view"
                    ], 8),
                    "dining": random.sample([
                        "Restaurant", "Bar", "Room service", "Continental breakfast",
                        "Buffet breakfast", "Cafe", "Lounge", "Mini-market"
                    ], random.randint(3, 6))
                },
                "photos": [
                    {"type": "exterior", "url": "https://example.com/hotel1.jpg"},
                    {"type": "lobby", "url": "https://example.com/lobby1.jpg"},
                    {"type": "room", "url": "https://example.com/room1.jpg"},
                    {"type": "restaurant", "url": "https://example.com/restaurant1.jpg"}
                ],
                "reviews": {
                    "overall_rating": round(random.uniform(7.5, 9.5), 1),
                    "total_reviews": random.randint(200, 3000),
                    "categories": {
                        "cleanliness": round(random.uniform(7.0, 9.5), 1),
                        "comfort": round(random.uniform(7.0, 9.5), 1),
                        "location": round(random.uniform(7.0, 9.5), 1),
                        "facilities": round(random.uniform(7.0, 9.5), 1),
                        "staff": round(random.uniform(7.0, 9.5), 1),
                        "value_for_money": round(random.uniform(6.5, 9.0), 1)
                    }
                }
            }
            
            return APIResponse(success=True, data=details)
            
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))


# Unified API manager
class TravelAPIManager:
    """Manages all travel-related APIs"""
    
    def __init__(self):
        self.weather_api = None
        self.currency_api = CurrencyAPIClient()
        self.news_api = None
        self.flight_api = MockFlightAPI()
        self.hotel_api = MockHotelAPI()
        
        # Initialize real APIs if keys are available
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API clients with available keys"""
        # Weather API
        weather_key = os.getenv("WEATHER_API_KEY")
        if weather_key:
            self.weather_api = WeatherAPIClient(weather_key)
        
        # News API
        news_key = os.getenv("NEWS_API_KEY")
        if news_key:
            self.news_api = NewsAPIClient(news_key)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all APIs"""
        results = {}
        
        if self.weather_api:
            async with self.weather_api:
                results["weather"] = await self.weather_api.health_check()
        else:
            results["weather"] = False
        
        # Currency API doesn't need auth, so always check
        currency_response = await self.currency_api.get_exchange_rates()
        results["currency"] = currency_response.success
        
        # Mock APIs are always "healthy"
        results["flights"] = True
        results["hotels"] = True
        
        return results
    
    async def get_comprehensive_weather(self, city: str, country_code: str = "") -> APIResponse:
        """Get comprehensive weather information"""
        if not self.weather_api:
            return APIResponse(
                success=False,
                data=None,
                error="Weather API not configured. Please set WEATHER_API_KEY."
            )
        
        async with self.weather_api:
            current = await self.weather_api.get_current_weather(city, country_code)
            if not current.success:
                return current
            
            forecast = await self.weather_api.get_forecast(city, 5, country_code)
            
            return APIResponse(
                success=True,
                data={
                    "current": current.data,
                    "forecast": forecast.data if forecast.success else None
                }
            )


# Example usage
async def test_apis():
    """Test the API integrations"""
    print("🔧 Testing API Integrations")
    print("=" * 40)
    
    manager = TravelAPIManager()
    
    # Health check
    health = await manager.health_check()
    print(f"API Health Status: {health}")
    
    # Test currency conversion
    currency_result = await manager.currency_api.convert_currency(100, "USD", "EUR")
    if currency_result.success:
        print(f"Currency: $100 USD = €{currency_result.data['converted_amount']} EUR")
    
    # Test mock flight search
    flight_result = await manager.flight_api.search_flights("New York", "London", "2024-06-15")
    if flight_result.success:
        print(f"Flights: Found {len(flight_result.data['flights'])} flights")
    
    # Test mock hotel search
    hotel_result = await manager.hotel_api.search_hotels("London", "2024-06-15", "2024-06-18")
    if hotel_result.success:
        print(f"Hotels: Found {len(hotel_result.data['hotels'])} hotels")


if __name__ == "__main__":
    asyncio.run(test_apis())
