export const fetchFile = async (file) => {
    const url = `./data/${file}`
    const raw = await fetch(url)

    return await raw.text()
}