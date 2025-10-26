"""
Generate 100 combinations of improvement techniques (v3 to v102)

Strategy: Greedy progressive enhancement
1. Test each technique individually (v3-v12)
2. Find best, then test best + each other technique (v13-v22)
3. Continue building on best combinations
"""
import json
import itertools
from typing import List, Dict, Tuple


def generate_combinations() -> List[Dict]:
    """
    Generate 100 version combinations using greedy strategy

    Returns:
        List of version configs from v3 to v102
    """
    techniques = list(range(1, 11))  # 10 techniques numbered 1-10
    combinations = []

    # ========================================
    # Phase 1: Single techniques (v3-v12)
    # ========================================
    for tech_id in techniques:
        combinations.append({
            'version': len(combinations) + 3,  # Start from v3
            'techniques': [tech_id],
            'description': f'Single technique: {tech_id}',
            'parent': 'v2',
        })

    # ========================================
    # Phase 2: All 2-combinations (v13-v57)
    # ========================================
    for pair in itertools.combinations(techniques, 2):
        combinations.append({
            'version': len(combinations) + 3,
            'techniques': sorted(list(pair)),
            'description': f'Pair: {pair}',
            'parent': 'v2',
        })

    # ========================================
    # Phase 3: Selected 3-combinations (v58-v92)
    # ========================================
    # Top 5 techniques based on projected gain: [1, 2, 3, 4, 5]
    top_techniques = [1, 2, 3, 4, 5]

    # All 3-combinations of top 5
    for triplet in itertools.combinations(top_techniques, 3):
        combinations.append({
            'version': len(combinations) + 3,
            'techniques': sorted(list(triplet)),
            'description': f'Top triplet: {triplet}',
            'parent': 'best_pair',
        })

    # Add some diverse 3-combinations (mix top with others)
    other_techniques = [6, 7, 8, 9, 10]
    for i, top_tech in enumerate(top_techniques[:5]):
        for other_tech in other_techniques[:2]:
            for third_tech in techniques:
                if third_tech != top_tech and third_tech != other_tech:
                    combo = sorted([top_tech, other_tech, third_tech])
                    if combo not in [c['techniques'] for c in combinations]:
                        combinations.append({
                            'version': len(combinations) + 3,
                            'techniques': combo,
                            'description': f'Mixed triplet: {combo}',
                            'parent': 'best_single',
                        })
                        if len(combinations) >= 92 - 3 + 1:
                            break
                if len(combinations) >= 92 - 3 + 1:
                    break
            if len(combinations) >= 92 - 3 + 1:
                break
        if len(combinations) >= 92 - 3 + 1:
            break

    # ========================================
    # Phase 4: 4-combinations (v93-v102)
    # ========================================
    # Best 4-combinations from top techniques
    for quad in itertools.combinations(top_techniques, 4):
        if len(combinations) >= 100:
            break
        combinations.append({
            'version': len(combinations) + 3,
            'techniques': sorted(list(quad)),
            'description': f'Top quad: {quad}',
            'parent': 'best_triplet',
        })

    # Add kitchen sink combinations
    while len(combinations) < 100:
        # All top 5
        if [1, 2, 3, 4, 5] not in [c['techniques'] for c in combinations]:
            combinations.append({
                'version': len(combinations) + 3,
                'techniques': [1, 2, 3, 4, 5],
                'description': 'All top 5 techniques',
                'parent': 'best_quad',
            })

        # Top 6
        if [1, 2, 3, 4, 5, 7] not in [c['techniques'] for c in combinations]:
            combinations.append({
                'version': len(combinations) + 3,
                'techniques': [1, 2, 3, 4, 5, 7],
                'description': 'Top 5 + EMA',
                'parent': 'best_combo',
            })

        # All except one
        for excluded in [10, 9, 8]:
            combo = [t for t in techniques if t != excluded]
            if combo not in [c['techniques'] for c in combinations] and len(combinations) < 100:
                combinations.append({
                    'version': len(combinations) + 3,
                    'techniques': combo,
                    'description': f'All except technique {excluded}',
                    'parent': 'best_combo',
                })

        # Kitchen sink (all 10)
        if techniques not in [c['techniques'] for c in combinations] and len(combinations) < 100:
            combinations.append({
                'version': len(combinations) + 3,
                'techniques': techniques,
                'description': 'Kitchen sink: all 10 techniques',
                'parent': 'best_combo',
            })

        # Safety: avoid infinite loop
        if len(combinations) >= 100:
            break

    # Trim to exactly 100
    combinations = combinations[:100]

    # Assign final version numbers
    for i, combo in enumerate(combinations):
        combo['version'] = i + 3  # v3 to v102

    return combinations


def save_combinations(combinations: List[Dict], output_path: str):
    """Save combinations to JSON"""
    with open(output_path, 'w') as f:
        json.dump(combinations, f, indent=2)

    print(f"✅ Generated {len(combinations)} combinations")
    print(f"   Saved to: {output_path}")

    # Statistics
    single = sum(1 for c in combinations if len(c['techniques']) == 1)
    pairs = sum(1 for c in combinations if len(c['techniques']) == 2)
    triplets = sum(1 for c in combinations if len(c['techniques']) == 3)
    quads = sum(1 for c in combinations if len(c['techniques']) == 4)
    more = sum(1 for c in combinations if len(c['techniques']) > 4)

    print(f"\n📊 Combination Statistics:")
    print(f"   Single techniques: {single}")
    print(f"   Pairs: {pairs}")
    print(f"   Triplets: {triplets}")
    print(f"   Quads: {quads}")
    print(f"   5+: {more}")


def generate_execution_plan(combinations: List[Dict], num_gpus: int = 4, jobs_per_gpu: int = 2):
    """
    Generate execution plan for multi-GPU training

    Returns:
        execution_plan: List of batches, each batch has up to num_gpus*jobs_per_gpu jobs
    """
    total_slots = num_gpus * jobs_per_gpu
    batches = []

    for i in range(0, len(combinations), total_slots):
        batch = combinations[i:i + total_slots]
        batches.append(batch)

    print(f"\n🔧 Execution Plan:")
    print(f"   Total combinations: {len(combinations)}")
    print(f"   GPUs: {num_gpus}")
    print(f"   Jobs per GPU: {jobs_per_gpu}")
    print(f"   Parallel slots: {total_slots}")
    print(f"   Total batches: {len(batches)}")
    print(f"   Jobs per batch: {total_slots}")

    return batches


def main():
    """Generate and save combinations"""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate combinations
    combinations = generate_combinations()

    # Save
    output_path = os.path.join(script_dir, 'combinations.json')
    save_combinations(combinations, output_path)

    # Generate execution plan
    execution_plan = generate_execution_plan(combinations, num_gpus=4, jobs_per_gpu=2)

    # Save execution plan
    plan_path = os.path.join(script_dir, 'execution_plan.json')
    with open(plan_path, 'w') as f:
        json.dump({'batches': execution_plan}, f, indent=2)

    print(f"\n💾 Saved execution plan to: {plan_path}")

    # Preview first batch
    print(f"\n🔍 First Batch Preview:")
    for i, combo in enumerate(execution_plan[0]):
        gpu_id = i // 2
        slot = i % 2
        print(f"   GPU {gpu_id} Slot {slot}: v{combo['version']} - {combo['description']}")


if __name__ == '__main__':
    main()
