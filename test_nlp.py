import sys
sys.path.append('.')
from utils.cast_lookup import CastLookup
c = CastLookup()
text = """Shadow Protocol is a high-stakes action thriller centered on a covert intelligence officer who uncovers a conspiracy involving autonomous weapons and a rogue multinational network. As loyalties collapse and time runs out, he must choose between saving his country or exposing the truth that could destabilize the world order.

Designed as a grounded but cinematic action film, the project balances character-driven storytelling with large-scale set pieces, practical effects, and emotional stakes. The narrative explores surveillance, morality in warfare, and the cost of secrecy, giving the film both commercial appeal and thematic depth.

Cast

Leonardo DiCaprio – Lead operative

Emily Blunt – Intelligence analyst and strategic partner

John Boyega – Field agent with conflicting loyalties

Ken Watanabe – International diplomat tied to the conspiracy

Crew

Director: Christopher Nolan

Writer: Taylor Sheridan

Cinematography: Roger Deakins

Music: Hans Zimmer

Production Company: Atlas Crown Studios

Positioning: Prestige action blockbuster with awards potential and strong global marketability."""
print(c.extract_names_from_text(text))
