import asyncio
from playwright.async_api import async_playwright
import os


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # Convert path to a proper file:// URI
        abs_path = os.path.abspath("docs/phase2_visualizations/tomtom_results.html")
        file_url = f"file:///{abs_path.replace(chr(92), '/')}"

        print(f"Loading {file_url}...")
        await page.goto(file_url, wait_until="networkidle", timeout=60000)

        print("Exporting to PDF...")
        await page.pdf(
            path="docs/phase2_visualizations/tomtom_results.pdf",
            format="A4",
            print_background=True,
            margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"},
        )
        await browser.close()
        print("PDF generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
