from flask import Flask, request, send_file, redirect, url_for, flash, render_template, send_from_directory
from flask_caching import Cache
import os
import pandas as pd
import random
from deap import base, creator, tools
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
def index():

    return render_template('index.html')

@cache.cached(timeout=300)  # Adjust timeout as needed
def send_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(f"File saved to {filename}")

        # Scan file with MetaDefender
        try:
            report = scan_file_with_metadefender(filename)
            print(f"MetaDefender report: {report}")

            # Check if the file is flagged by any antivirus
            if any(result.get('scan_result') == 'malicious' for result in report.get('data', {}).get('scan_results', [])):
                flash('File is infected with malware')
                os.remove(filename)
                return redirect(url_for('index'))
            
            # Proceed with optimization if file is clean
            optimized_rules = optimize_firewall_rules(filename)
            optimized_file = 'optimized_rules.csv'
            save_optimized_rules(optimized_rules, optimized_file)
            
            # Automatically download the optimized file
            return send_file(optimized_file, as_attachment=True)

        except Exception as e:
            flash(f"Error processing file: {e}")
            os.remove(filename)
            return redirect(url_for('index'))

def scan_file_with_metadefender(file_path):
    METAEDEFENDER_API_KEY = '0b0f9fa8f8e58967b7f3720603fb8d3a'
    METAEDEFENDER_URL = 'https://api.metadefender.com/v4/file'
    headers = {'apikey': METAEDEFENDER_API_KEY}
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(METAEDEFENDER_URL, headers=headers, files=files)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise Exception(f"Error uploading file: {response.status_code} - {response.text}")

def optimize_firewall_rules(file_path):
    df = pd.read_csv(file_path)

    def evaluate(individual):
        score = sum(individual)
        return (score,)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(df))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population_size = 100
    num_generations = 50
    crossover_prob = 0.7
    mutation_prob = 0.2

    def convert_to_firewall_rules(binary_repr):
        firewall_rules = []
        for idx, allow_flag in enumerate(binary_repr):
            if allow_flag == 1:
                port_number = idx + 1
                rule = f"Allow TCP traffic on port {port_number}"
                firewall_rules.append(rule)
        return firewall_rules

    population = toolbox.population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(num_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print(f"Gen: {gen}, Min: {min(fits)}, Max: {max(fits)}, Avg: {mean}, Std: {std}")

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    firewall_rules = convert_to_firewall_rules(best_ind)
    return pd.DataFrame(firewall_rules, columns=["Rule"])

def save_optimized_rules(rules, file_path):
    rules.to_csv(file_path, index=False)

if __name__ == '__main__':
    app.run(debug=True)
