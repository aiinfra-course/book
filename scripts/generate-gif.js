const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function generateGif(htmlFile, outputGif, steps = 13, stepDelay = 1000) {
    const outputDir = path.join(__dirname, '../.temp');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-gpu',
            '--window-size=1200,800',
        ],
        defaultViewport: { width: 1200, height: 800 },
    });

    const page = await browser.newPage();
    const fileUrl = `file://${path.resolve(__dirname, '../public', htmlFile)}`;
    await page.goto(fileUrl, { waitUntil: 'networkidle0' });

    await delay(2000);

    for (let i = 0; i < steps; i++) {
        const screenshotPath = path.join(outputDir, `step-${String(i).padStart(3, '0')}.png`);
        await page.screenshot({ path: screenshotPath, fullPage: false });
        console.log(`Step ${i + 1}/${steps}: ${screenshotPath}`);

        if (i < steps - 1) {
            await page.evaluate(() => {
                if (typeof nextStep === 'function') {
                    nextStep();
                }
            });
            await delay(stepDelay);
        }
    }

    await browser.close();

    const palettePath = path.join(outputDir, 'palette.png');
    execSync(`ffmpeg -i ${outputDir}/step-%03d.png -filter_complex "[0:v] palettegen" ${palettePath}`);
    
    execSync(`ffmpeg -framerate 1 -i ${outputDir}/step-%03d.png -i ${palettePath} -filter_complex "[0:v][1:v] paletteuse" ${outputGif}`);
    
    fs.rmSync(outputDir, { recursive: true, force: true });

    console.log(`GIF generated: ${outputGif}`);
}

async function main() {
    const htmlFile = 'demo.html';
    const outputGif = path.join(__dirname, '../public/demo.gif');
    
    await generateGif(htmlFile, outputGif);
}

main().catch(console.error);
