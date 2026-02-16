import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import ClaimsListPage from "./pages/ClaimsListPage";
import PractitionerPage from "./pages/PractitionerPage";
import SubmitClaimPage from "./pages/SubmitClaimPage";

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="app-title">Claims Dashboard</h1>
        <nav>
          <NavLink className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")} to="/submit">
            Submit Claim
          </NavLink>
          <NavLink className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")} to="/claims">
            All Claims
          </NavLink>
          <NavLink className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")} to="/practitioner">
            Practitioner View
          </NavLink>
        </nav>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<Navigate to="/submit" replace />} />
          <Route path="/submit" element={<SubmitClaimPage />} />
          <Route path="/claims" element={<ClaimsListPage />} />
          <Route path="/practitioner" element={<PractitionerPage />} />
          <Route path="/practitioner/:submissionId" element={<PractitionerPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
