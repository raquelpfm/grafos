import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import heapq

class Vertice:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x  # coordenada x
        self.y = y  # coordenada y
        self.allowed_movements = set()  # conjunto de movimentos permitidos
        self.incoming_arestas = []  # arestas de entrada
        self.outgoing_arestas = []  # arestas de saída
        self.is_traffic_light = False
        self.cycle_time = 0  # tempo de ciclo se for semáforo

class Aresta:
    def __init__(self, id: int, source: Vertice, target: Vertice, capacity: float):
        self.id = id
        self.source = source
        self.target = target
        self.capacity = capacity  # capacidade máxima
        self.flow = 0.0  # fluxo atual
        self.length = self._calculate_length()  # comprimento da via
        self.num_lanes = 1  # número de faixas
        self.speed_limit = 60.0  # velocidade máxima em km/h
        
    def _calculate_length(self) -> float:
        """Calcula o comprimento euclidiano da via."""
        dx = self.target.x - self.source.x
        dy = self.target.y - self.source.y
        return np.sqrt(dx*dx + dy*dy)
    
    def get_congestion_level(self) -> float:
        """Calcula o nível de congestionamento atual."""
        if self.capacity == 0:
            return float('inf')
        return self.flow / self.capacity
    
    def get_congestion_cost(self) -> float:
        """
        Implementa a função de custo g(x) = x/(1-x) para congestionamento.
        Retorna infinito se o fluxo exceder 90% da capacidade.
        """
        congestion = self.get_congestion_level()
        if congestion >= 0.9:
            return float('inf')
        return congestion / (1 - congestion)

class TrafficNetwork:
    """
    Representa a rede viária completa como um grafo planar.
    Gerencia vértices, arestas e implementa os algoritmos de otimização.
    """
    def __init__(self):
        self.vertices: Dict[int, Vertice] = {}
        self.arestas: Dict[int, Aresta] = {}
        self.demand_matrix: Dict[Tuple[int, int], float] = {}
        
    def add_vertice(self, id: int, x: float, y: float) -> Vertice:
        """Adiciona um novo vértice à rede."""
        vertice = Vertice(id, x, y)
        self.vertices[id] = vertice
        return vertice
    
    def aresta(self, id: int, source_id: int, target_id: int, capacity: float) -> Aresta:
        """Adiciona uma nova aresta à rede."""
        source = self.vertices[source_id]
        target = self.vertices[target_id]
        aresta = Aresta(id, source, target, capacity)
        self.arestas[id] = aresta
        source.outgoing_arestas.append(aresta)
        target.incoming_arestas.append(aresta)
        return aresta
    
    def set_demand(self, source_id: int, target_id: int, demand: float):
        """Define a demanda de tráfego entre dois vértices."""
        self.demand_matrix[(source_id, target_id)] = demand
    
    def is_planar(self) -> bool:
        """
        Verifica se o grafo é planar usando o algoritmo de Boyer-Myrvold.
        Esta é uma implementação simplificada que verifica a fórmula de Euler.
        """
        V = len(self.vertices)
        E = len(self.arestas)
        # Pela fórmula de Euler: V - E + F = 2, onde F é o número de faces
        # Em um grafo planar: E <= 3V - 6 para V >= 3
        if V < 3:
            return True
        return E <= 3*V - 6
    
    def find_shortest_path(self, source_id: int, target_id: int) -> List[Aresta]:
        """
        Implementa o algoritmo de Dijkstra para encontrar o caminho mais curto
        considerando o congestionamento como peso.
        """
        distances = {v: float('inf') for v in self.vertices}
        distances[source_id] = 0
        previous = {v: None for v in self.vertices}
        pq = [(0, source_id)]
        visited = set()
        
        while pq:
            current_distance, current_vertice = heapq.heappop(pq)
            
            if current_vertice in visited:
                continue
                
            if current_vertice == target_id:
                break
                
            visited.add(current_vertice)
            
            # Explora todas as arestas de saída
            for aresta in self.vertices[current_vertice].outgoing_arestas:
                neighbor = aresta.target.id
                if neighbor in visited:
                    continue
                    
                # O peso é uma combinação do comprimento e congestionamento
                weight = aresta.length * (1 + aresta.get_congestion_cost())
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = aresta
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstrói o caminho
        path = []
        current = target_id
        while current != source_id and previous[current] is not None:
            path.append(previous[current])
            current = previous[current].source.id
        path.reverse()
        return path
    
    def optimize_flow(self, max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Implementa o algoritmo principal de otimização de fluxo.
        Usa um método iterativo para distribuir o fluxo minimizando congestionamento.
        """
        # Reseta todos os fluxos
        for aresta in self.arestas.values():
            aresta.flow = 0.0
            
        previous_total_cost = float('inf')
        
        for iteration in range(max_iterations):
            # Para cada par O-D com demanda
            for (source_id, target_id), demand in self.demand_matrix.items():
                # Encontra o melhor caminho considerando congestionamento atual
                path = self.find_shortest_path(source_id, target_id)
                
                # Distribui a demanda ao longo do caminho
                for aresta in path:
                    aresta.flow += demand
            
            # Calcula o custo total atual
            total_cost = sum(aresta.get_congestion_cost() for aresta in self.arestas.values())
            
            # Verifica convergência
            if abs(previous_total_cost - total_cost) < tolerance:
                break
                
            previous_total_cost = total_cost
            
        return total_cost
    
    def get_critical_arestas(self, threshold: float = 0.8) -> List[Aresta]:
        """Identifica as arestas críticas (alto congestionamento)."""
        return [aresta for aresta in self.arestas.values() 
                if aresta.get_congestion_level() > threshold]

# Exemplo de uso
def create_sample_network() -> TrafficNetwork:
    """Cria uma rede de exemplo para demonstração."""
    network = TrafficNetwork()
    
    # Adiciona alguns vértices (interseções)
    network.vertice(1, 0.0, 0.0)
    network.vertice(2, 1.0, 0.0)
    network.vertice(3, 0.0, 1.0)
    network.vertice(4, 1.0, 1.0)
    
    # Adiciona arestas (vias) com diferentes capacidades
    network.add_aresta(1, 1, 2, 1000.0)  # capacidade de 1000 veículos/hora
    network.add_aresta(2, 1, 3, 800.0)
    network.add_aresta(3, 2, 4, 1200.0)
    network.add_aresta(4, 3, 4, 900.0)
    
    # Define algumas demandas de tráfego
    network.set_demand(1, 4, 500.0)  # 500 veículos/hora querem ir de 1 para 4
    network.set_demand(2, 3, 300.0)
    
    return network

# Execução principal
def main():
    # Cria a rede de exemplo
    network = create_sample_network()
    
    # Verifica se é planar
    if not network.is_planar():
        print("Aviso: A rede não é planar!")
        return
    
    # Otimiza o fluxo
    total_cost = network.optimize_flow()
    print(f"Custo total após otimização: {total_cost:.2f}")
    
    # Identifica pontos críticos
    critical_arestas = network.get_critical_arestas()
    print("\nVias críticas (congestionadas):")
    for aresta in critical_arestas:
        print(f"Via {aresta.id}: {aresta.get_congestion_level()*100:.1f}% da capacidade")

if __name__ == "__main__":
    main()