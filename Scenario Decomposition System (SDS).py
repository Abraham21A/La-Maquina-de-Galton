import openai
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import plotly.graph_objects as go
import numpy as np

# Initialize OpenAI, spaCy, and download NLTK resources
openai.api_key = "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"
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')

def clean_and_tokenize(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text using NLTK's word_tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def get_group_discussion():
    return input("Group Leader: What group discussion would you like to initiate? ")

def get_user_preferences():
    destination = input("Enter your preferred destination: ")
    activities = input("What activities would you like to do? ")
    stay = input("Where would you like to stay? ")
    budget = input("What's your budget? ")
    days = input("How many days do you plan to travel? ")
    return destination, activities, stay, budget, days

def swot_analysis(component, details):
    question = f"Considering the {component} '{details}' for the trip, what are the Strengths, Weaknesses, Opportunities, and Threats?"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=question,
        max_tokens=500
    ).choices[0].text.strip()

    strengths = extract_component_from_response(response, "Strengths")
    weaknesses = extract_component_from_response(response, "Weaknesses")
    opportunities = extract_component_from_response(response, "Opportunities")
    threats = extract_component_from_response(response, "Threats")

    return {"Strengths": strengths, "Weaknesses": weaknesses, "Opportunities": opportunities, "Threats": threats}

def extract_component_from_response(response, component_name):
    component_pattern = re.compile(f'{component_name}: (.*?)(Strengths:|Weaknesses:|Opportunities:|Threats:|$)', re.DOTALL)
    match = component_pattern.search(response)
    return match.group(1).strip() if match else "N/A"

def visualize_got_simple(group_discussion, user_details):
    G = nx.Graph()
    G.add_node("Group Decision")
    G.add_node("Group Leader")
    G.add_edge("Group Decision", "Group Leader")
    G.add_node(group_discussion)
    G.add_edge("Group Leader", group_discussion)

    for i, details in enumerate(user_details, 1):
        user_node = f"User_{i}"
        G.add_node(user_node)
        G.add_edge("Group Decision", user_node)

        for detail in details:
            G.add_node(detail)
            G.add_edge(user_node, detail)

    nx.draw(G, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", width=2, edge_color="gray")
    plt.show()

def visualize_got_3d(group_discussion, user_details):
    G = nx.Graph()
    G.add_node("Group Decision")
    G.add_node("Group Leader")
    G.add_edge("Group Decision", "Group Leader")
    G.add_node(group_discussion)
    G.add_edge("Group Leader", group_discussion)

    for i, details in enumerate(user_details, 1):
        user_node = f"User_{i}"
        G.add_node(user_node)
        G.add_edge("Group Decision", user_node)

        for detail in details:
            G.add_node(detail)
            G.add_edge(user_node, detail)

    # Create 3D positions for nodes
    pos = nx.spring_layout(G, dim=3, seed=42)

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(adjacencies[0])

    node_trace.marker.size = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        scene=dict(
                            xaxis=dict(nticks=4, range=[np.min(node_x), np.max(node_x)]),
                            yaxis=dict(nticks=4, range=[np.min(node_y), np.max(node_y)]),
                            zaxis=dict(nticks=4, range=[np.min(node_z), np.max(node_z)]),
                        ),
                        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                        method='animate', args=[None, dict(frame=dict(duration=100, redraw=True),
                                        fromcurrent=True)]), dict(label='Pause', method='animate',
                                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])])],

                    ))

    frames = [go.Frame(data=[go.Scatter3d(x=[j/100*node for node in node_x],
                                          y=[j/100*node for node in node_y],
                                          z=[j/100*node for node in node_z],
                                          mode="markers")]) for j in range(100)]

    fig.frames = frames

    fig.show()

def evaluate_swot_analysis(swot_responses):
    evaluations = []

    for swot in swot_responses:
        score = 0

        # Add scores for strengths and opportunities
        score += len(swot["Strengths"].split()) + len(swot["Opportunities"].split())

        # Subtract scores for weaknesses and threats
        score -= len(swot["Weaknesses"].split()) + len(swot["Threats"].split())

        evaluations.append(score)

    return evaluations

class ContextualInputParser:
    def __init__(self, context):
        self.context = context
        self.tokens = clean_and_tokenize(context)
        self.doc = nlp(context)

    def extract_entities(self):
        return [ent.text for ent in self.doc.ents]

    def extract_key_phrases(self):
        return [chunk.text for chunk in self.doc.noun_chunks]

class ComponentAnalyzer:
    def __init__(self, components):
        self.components = components

    def sentiment_analysis(self, component):
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"Sentiment of {component}?",
            max_tokens=5
        )
        return response.choices[0].text.strip()

    def cluster_components(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.components)
        kmeans = KMeans(n_clusters=2).fit(X)
        return kmeans.labels_

def display_swot_for_chosen_destination(destination, swot_responses, decision_points):
    index = decision_points.index(destination)
    swot = swot_responses[index]
    print("\nSWOT Analysis for chosen destination -", destination)
    print("Strengths:\n", swot["Strengths"])
    print("\nWeaknesses:\n", swot["Weaknesses"])
    print("\nOpportunities:\n", swot["Opportunities"])
    print("\nThreats:\n", swot["Threats"])

class DecisionComponentGenerator:
    def __init__(self, analyzed_components):
        self.analyzed_components = analyzed_components

    def generate_decision_elements(self):
        return [component + "_decision" for component in self.analyzed_components]

class IntegrationWithExternalSystems:
    def __init__(self, decision_elements):
        self.decision_elements = decision_elements

    def standardize_output(self):
        return {"decision_elements": self.decision_elements}

class ScenarioDecompositionSystem:
    def __init__(self, user_details):
        self.user_details = user_details

    def extract_entities(self):
        return [f"User_{i+1}" for i in range(len(self.user_details))]

    def extract_preferences(self):
        preferences = {}
        for i, details in enumerate(self.user_details):
            user = f"User_{i+1}"
            destination = details[0]  # Assuming first detail is the destination
            activities = details[1]   # Assuming second detail is the activities
            preferences[user] = f"{destination} - {activities}"
        return preferences

    def decision_points(self):
        return [details[0] for details in self.user_details]  # Assuming first detail is the destination

def analyze_scenario(group_discussion, user_details):
    combined_text = group_discussion + ' ' + ' '.join([' '.join(detail) for detail in user_details])
    cip = ContextualInputParser(combined_text)
    entities = ["Group Leader"] + [f"User_{i+1}" for i in range(len(user_details))] + cip.extract_entities()
    key_phrases = cip.extract_key_phrases()

    ca = ComponentAnalyzer(entities + key_phrases)
    sentiments = [ca.sentiment_analysis(component) for component in entities + key_phrases]
    clusters = ca.cluster_components()

    dcg = DecisionComponentGenerator(entities + key_phrases)
    decision_elements = dcg.generate_decision_elements()

    integration = IntegrationWithExternalSystems(decision_elements)
    standardized_output = integration.standardize_output()

    # Integrate ScenarioDecompositionSystem
    sds = ScenarioDecompositionSystem(user_details)
    sds_entities = sds.extract_entities()
    sds_preferences = sds.extract_preferences()
    sds_decision_points = sds.decision_points()

    sds_output = {
        "Entities": sds_entities,
        "Preferences": sds_preferences,
        "Decision Points": sds_decision_points
    }

    return standardized_output, sds_output

def decide_destination(evaluations, decision_points):
    # Determine the destination with the highest SWOT evaluation score.
    best_index = evaluations.index(max(evaluations))
    return decision_points[best_index]

def main():
    group_discussion = get_group_discussion()
    num_users = int(input("How many users are in the group? "))

    user_details = []
    for i in range(num_users):
        print(f"\nUser {i + 1} Preferences:")
        user_details.append(get_user_preferences())

    # Display the simple graph
    visualize_got_simple(group_discussion, user_details)

    # Display the 3D graph
    visualize_got_3d(group_discussion, user_details)

    analysis, sds_output = analyze_scenario(group_discussion, user_details)

    decision_points = sds_output["Decision Points"]
    swot_responses = [swot_analysis("destination", dp) for dp in decision_points]

    evaluations = evaluate_swot_analysis(swot_responses)
    final_decision = decide_destination(evaluations, decision_points)

    # Display the decision to the user
    print("\nBased on the SWOT analysis and preferences of the users, the best destination to choose is:", final_decision)
    display_swot_for_chosen_destination(final_decision, swot_responses, decision_points)


    print("\nScenario Decomposition Output:")
    for key, value in sds_output.items():
        print(key + ":")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"    {subkey} - {subvalue}")
        else:
            for item in value:
                print(f"    {item}")

if __name__ == "__main__":
    main()