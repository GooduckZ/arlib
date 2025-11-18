"""
Utility functions for bit-vector optimization.

This module provides helper functions for converting between different
representations used in bit-vector optimization problems.
"""
import logging
import os
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def cnt(result: List[int]) -> int:
    """Convert a list of binary digits to an integer.

    The list is interpreted as little-endian (LSB first after reversal).

    Args:
        result: List of integers representing binary digits (will be reversed)

    Returns:
        Integer value represented by the binary digits
    """
    result.reverse()
    total = 0
    for i, bit in enumerate(result):
        if bit > 0:
            total += 2 ** i
    return total


def list_to_int(result: List[List[int]], obj_type: List[int]) -> List[int]:
    """Convert lists of binary results to integers based on objective type.

    Args:
        result: List of binary result lists
        obj_type: List indicating objective type (0 for minimize, 1 for maximize)

    Returns:
        List of integer values, converted based on objective type
    """
    res: List[int] = []
    for i, binary_result in enumerate(result):
        score = cnt(binary_result)
        if obj_type[i] == 1:
            # Maximization: use score directly
            res.append(score)
        else:
            # Minimization: invert the score
            max_value = 2 ** len(binary_result) - 1
            res.append(max_value - score)
    return res


def assum_in_m(assum: List[int], m: List[int]) -> bool:
    """Check if all assumptions are in the model.

    Args:
        assum: List of assumption literals
        m: List of model literals

    Returns:
        True if all assumptions are in the model, False otherwise
    """
    return all(lit in m for lit in assum)


def cnf_from_z3(constraint_file: str) -> Optional[str]:
    """Generate CNF from Z3 constraint file.

    Args:
        constraint_file: Path to the constraint file

    Returns:
        Z3 output as string, or None if an error occurred
    """
    path = os.getcwd()
    path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    z3_path = os.path.join(path, 'z3', 'build', 'z3')
    try:
        command = [z3_path, "opt.priority=box", constraint_file]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Error running Z3: %s", e)
        return None


def read_cnf(data: str) -> Optional[Tuple[List[List[int]], List[List[int]], List[int]]]:
    """Parse CNF data from Z3 output.

    Args:
        data: String containing CNF data from Z3

    Returns:
        Tuple of (clauses, soft_clauses, obj_type) where:
        - clauses: List of hard clauses
        - soft_clauses: List of soft clause groups
        - obj_type: List indicating objective type (0 for minimize, 1 for maximize)
        Returns None if parsing fails
    """
    lines = data.splitlines()

    clauses: List[List[int]] = []
    obj_type: List[int] = []  # 0 for minimize, 1 for maximize
    soft: List[List[int]] = []
    constraint_type = '0'
    soft_temp: List[int] = []

    # Parse first line to get number of clauses
    first_line = lines[0].strip()
    parts = first_line.split()
    if len(parts) < 4:
        logger.error("Invalid CNF header: %s", first_line)
        return None
    num_clauses = int(parts[3])

    # Parse clauses
    i = 1
    while i <= num_clauses:
        clause = list(map(int, lines[i].split()))
        clause.pop()  # Remove trailing 0
        clauses.append(clause)
        i += 1

    # Parse comment lines for soft clauses
    j = i
    comment_dict: dict[int, str] = {}
    min_index = 10 ** 10
    while j < len(lines) and lines[j].startswith('c'):
        line_parts = lines[j].split()
        if len(line_parts) < 6:
            j += 1
            continue
        split_by_excl = lines[j].split('!')
        try:
            index = int(split_by_excl[-1])
            comment_dict[index] = lines[j]
            if index < min_index:
                min_index = index
        except (ValueError, IndexError) as e:
            logger.error("Error parsing comment line %s: %s", lines[j], e)
            return None
        j += 1

    # Reorder comment lines
    for k, line in comment_dict.items():
        lines[i + k - min_index] = line

    # Parse soft clauses
    num_comments = len(comment_dict)
    for k in range(num_comments):
        parts = lines[i + k].split()
        if len(parts) < 6:
            break

        if parts[4].endswith(':0]'):
            # End of current soft clause group
            if soft_temp:
                soft.append(soft_temp)
                soft_temp = []
                obj_type.append(int(constraint_type))
        constraint_type = parts[3][3]
        soft_temp.append(int(parts[1]))

    # Handle remaining soft clauses
    if soft_temp:
        soft.append(soft_temp)
        obj_type.append(int(constraint_type))

    return clauses, soft, obj_type


def res_z3_trans(r_z3: str) -> List[int]:
    """Extract results from Z3 output.

    Args:
        r_z3: String containing Z3 output

    Returns:
        List of integer results extracted from Z3 output
    """
    lines = r_z3.splitlines()
    results: List[int] = []
    # Skip first two lines (header)
    for line in lines[2:]:
        parts = line.split()
        if len(parts) > 1:
            # Remove trailing character (usually ':')
            value_str = parts[1][:-1]
            try:
                results.append(int(value_str))
            except ValueError:
                logger.warning("Could not parse value from line: %s", line)
    return results


if __name__ == '__main__':
    benchmark_path = '/arlib/benchmarks/omt/'
    result = subprocess.run(
        ['z3', 'opt.priority=box', benchmark_path],
        capture_output=True,
        text=True,
        check=True
    )
    print(res_z3_trans(result.stdout))
