#!/usr/bin/env python3
"""
Test script to verify environment configuration is working correctly.
Run this after setting up your .env file.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

def test_config():
    """Test configuration loading and validation."""
    print("\n" + "="*70)
    print("REASON Configuration Test")
    print("="*70 + "\n")

    try:
        from configs.env_config import config
        print("Successfully imported configuration module\n")
    except Exception as e:
        print(f"Failed to import configuration: {e}")
        return False

    # Display configuration
    print("Current Configuration:")
    print("-" * 70)
    config.display()

    # Validate configuration
    print("\nValidating Configuration:")
    print("-" * 70)
    is_valid = config.validate()

    if is_valid:
        print("\nConfiguration is valid!")
    else:
        print("\nConfiguration has warnings (see above)")
        print("   Update your .env file to resolve them.\n")

    # Test specific values
    print("\nTesting Specific Values:")
    print("-" * 70)

    tests = [
        ("GCP_PROJECT_ID", config.GCP_PROJECT_ID),
        ("GCP_LOCATION", config.GCP_LOCATION),
        ("NCBI_EMAIL", config.NCBI_EMAIL),
        ("LLM_TEMPERATURE", config.LLM_TEMPERATURE),
        ("LLM_MAX_TOKENS", config.LLM_MAX_TOKENS),
        ("DEFAULT_MAX_STUDIES", config.DEFAULT_MAX_STUDIES),
        ("PUBMED_MAX_RESULTS", config.PUBMED_MAX_RESULTS),
        ("ARTIFACTS_PICO_DIR", config.ARTIFACTS_PICO_DIR),
    ]

    all_good = True
    for name, value in tests:
        if value:
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: Not set!")
            all_good = False

    # Test imports from other modules
    print("\nTesting Module Integration:")
    print("-" * 70)

    try:
        from configs.config import PROJECT_ID, LOCATION, ENDPOINT_ID
        print(f"  configs.config imports working")
        print(f"     PROJECT_ID: {PROJECT_ID}")
    except Exception as e:
        print(f"  configs.config import failed: {e}")
        all_good = False

    try:
        from pipeline.extractor.pico_extractor import _agent
        print(f"  pico_extractor._agent() can be imported")
    except Exception as e:
        print(f"  pico_extractor import failed: {e}")
        all_good = False

    # Summary
    print("\n" + "="*70)
    if all_good and is_valid:
        print("All tests passed! Configuration is ready to use.")
    else:
        print("Some tests failed or have warnings.")
        print("   Please review the output above and update your .env file.")
    print("="*70 + "\n")

    return all_good and is_valid


if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
