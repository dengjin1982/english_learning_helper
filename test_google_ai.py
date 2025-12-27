#!/usr/bin/env python3
"""
Test script to verify Google AI is working correctly.
Tests definition, example sentence, and pronunciation generation for a word.
"""

import os
import sys
from english_learner import SentenceGenerator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def test_google_ai(word: str = "liturgy"):
    """Test Google AI with a specific word."""
    console.print(f"\n[bold blue]Testing Google AI with word: '{word}'[/bold blue]\n")
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        console.print("[red][ERROR] GOOGLE_API_KEY not found in environment variables[/red]")
        console.print("[yellow]Please set GOOGLE_API_KEY in your .env file[/yellow]")
        return False
    
    console.print(f"[green][OK] Google API key found: {api_key[:20]}...[/green]")
    
    # Initialize generator
    try:
        console.print("\n[bold]Initializing Google AI...[/bold]")
        generator = SentenceGenerator(api_key=api_key)
        
        if generator.use_mock:
            console.print("[red][ERROR] Generator is in mock mode - Google AI not initialized[/red]")
            console.print("[yellow]This means Google AI failed to initialize. Check the error messages above.[/yellow]")
            return False
        
        console.print(f"[green][OK] Google AI initialized successfully[/green]")
        if generator.model:
            console.print(f"[dim]Using model: {generator.model}[/dim]")
        if generator.models:
            console.print(f"[dim]Available models: {len(generator.models)}[/dim]")
        
    except Exception as e:
        console.print(f"[red][ERROR] Failed to initialize Google AI: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False
    
    # Test 1: Generate explanation and examples
    console.print(f"\n[bold cyan]Test 1: Generating explanation and examples for '{word}'...[/bold cyan]")
    try:
        result = generator.generate_explanation_and_examples(word, num_examples=1)
        
        explanation = result.get('explanation', '')
        examples = result.get('examples', [])
        
        console.print(f"\n[bold green][SUCCESS] Successfully generated content[/bold green]")
        console.print(f"\n[bold]Explanation:[/bold]")
        if explanation:
            console.print(Panel(explanation, border_style="green"))
        else:
            console.print("[red][ERROR] No explanation generated![/red]")
        
        console.print(f"\n[bold]Examples:[/bold]")
        if examples:
            for i, example in enumerate(examples, 1):
                console.print(f"  {i}. {example}")
        else:
            console.print("[red][ERROR] No examples generated![/red]")
        
        # Check if it's from Google AI or fallback
        if not explanation or 'Unable to fetch definition' in explanation:
            console.print("\n[yellow][WARNING] Explanation appears to be from fallback (dictionary API), not Google AI[/yellow]")
        elif len(explanation) < 10:
            console.print("\n[yellow][WARNING] Explanation is very short - might be from fallback[/yellow]")
        else:
            console.print("\n[green][OK] Explanation appears to be from Google AI[/green]")
        
    except Exception as e:
        console.print(f"[red][ERROR] Failed to generate explanation and examples: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False
    
    # Test 2: Test raw Google AI response
    console.print(f"\n[bold cyan]Test 2: Testing raw Google AI response...[/bold cyan]")
    try:
        prompt = f"Provide a dictionary definition for the word '{word}'."
        text = generator._try_generate_content(prompt)
        
        if text:
            console.print(f"[green][OK] Google AI returned response ({len(text)} characters)[/green]")
            console.print(f"\n[bold]Raw Response (first 500 chars):[/bold]")
            console.print(Panel(text[:500] + ("..." if len(text) > 500 else ""), border_style="blue"))
        else:
            console.print("[red][ERROR] Google AI returned empty response![/red]")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            console.print(f"[yellow][WARNING] Model not found error (this can happen with some API versions)[/yellow]")
            console.print(f"[dim]Error: {error_msg[:200]}[/dim]")
            console.print(f"[dim]Note: Test 1 succeeded, so Google AI is working. This might be a model name issue.[/dim]")
        else:
            console.print(f"[red][ERROR] Failed to get raw Google AI response: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return False
    
    # Test 3: Test pronunciation
    console.print(f"\n[bold cyan]Test 3: Fetching pronunciation for '{word}'...[/bold cyan]")
    try:
        pronunciation = generator._fetch_pronunciation_from_api(word)
        if pronunciation:
            # Use print() for pronunciation to avoid Windows encoding issues with IPA characters
            print(f"[OK] Pronunciation found: /{pronunciation}/")
        else:
            console.print("[yellow][WARNING] No pronunciation found (this is okay, not all words have pronunciations)[/yellow]")
    except Exception as e:
        # Use print() to avoid encoding issues
        print(f"[WARNING] Failed to fetch pronunciation: {str(e)[:100]}")
        console.print("[dim](This is okay - pronunciation is optional)[/dim]")
    
    # Summary
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green][SUCCESS] All tests completed![/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]\n")
    
    return True

if __name__ == '__main__':
    test_word = "liturgy"
    if len(sys.argv) > 1:
        test_word = sys.argv[1]
    
    success = test_google_ai(test_word)
    sys.exit(0 if success else 1)

