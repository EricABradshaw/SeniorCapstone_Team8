import './index.scss'
function Home () {
  return (
    <div className="Home">
      <div id="app-header">
        <h1>StegoSource</h1>
      </div>
      <div id="mainSection">
        <div className="filler"></div>
        <button >Generate Stego Image</button>
        <button >Extract Secret Image</button>
        <button >About</button>
        <div className="filler"></div>
      </div>
    </div>
  )
}
export default Home